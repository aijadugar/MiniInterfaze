"""
Mini-Interfaze — Receipt Field Extractor
Standalone inference app for Hugging Face Spaces (Gradio).

Loads a checkpoint trained in the companion notebook (CRNN + adapter + decoder) and
runs the full pipeline on an uploaded receipt photo:
  pretrained CRAFT detector (EasyOCR) -> CRNN recognizer -> adapter projection
  -> transformer decoder -> structured JSON + precontext (boxes/confidence)

This file is self-contained: no imports from the training notebook are required.
The custom CUDA kernel used during training is NOT required for inference — the
FusedRMSNormResidual module below automatically falls back to a numerically
identical pure-PyTorch implementation when no CUDA device is available, which is
the normal case on a free CPU-tier Space.

Checkpoint format: safetensors, not torch.save/.pt. The model repo
(https://huggingface.co/aijadugar/receipt-field-extractor) stores two files:
  - model.safetensors  (flat {"crnn.<key>": tensor, "adapter.<key>": tensor,
                                "decoder.<key>": tensor})
  - config.json         (non-tensor metadata: dims, vocab, field names)
If those two files aren't sitting next to app.py in the Space repo, they're
pulled automatically from the Hub the first time load_models() runs.
"""

import os
import json
import html
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import gradio as gr
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Config / constants (must match the training notebook)
# ---------------------------------------------------------------------------
HF_REPO_ID = "aijadugar/receipt-field-extractor"
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHARS = string.ascii_uppercase + string.digits + " .,:/-'&"
BLANK_IDX = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
CRNN_VOCAB_SIZE = len(CHARS) + 1

SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[SEP]"]
CHARSET = list(string.ascii_uppercase + string.digits + " .,:/-'&")
DEC_VOCAB = SPECIAL + CHARSET
DEC_TOK2IDX = {t: i for i, t in enumerate(DEC_VOCAB)}
DEC_IDX2TOK = {i: t for i, t in enumerate(DEC_VOCAB)}
DEC_VOCAB_SIZE = len(DEC_VOCAB)
PAD, BOS, EOS, SEP = (DEC_TOK2IDX[t] for t in SPECIAL)

FIELDS = ["company", "date", "address", "total"]
FIELD_ICONS = {"company": "🏬", "date": "📅", "address": "📍", "total": "💰"}


def ctc_greedy_decode(logits):
    probs = logits.softmax(-1)
    conf, idx = probs.max(-1)
    idx, conf = idx.tolist(), conf.tolist()
    out_chars, out_conf, prev = [], [], None
    for i, c in zip(idx, conf):
        if i != prev and i != BLANK_IDX:
            out_chars.append(IDX2CHAR.get(i, ""))
            out_conf.append(c)
        prev = i
    text = "".join(out_chars)
    mean_conf = float(np.mean(out_conf)) if out_conf else 0.0
    return text, mean_conf


# ---------------------------------------------------------------------------
# Model definitions (identical to the training notebook)
# ---------------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, img_h=32, vocab_size=CRNN_VOCAB_SIZE, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        feat_h = img_h // 8
        self.rnn = nn.LSTM(256 * feat_h, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def hidden_states(self, x):
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, W, C * H)
        out, _ = self.rnn(feat)
        return out

    def forward(self, x):
        return self.fc(self.hidden_states(x))


class FusedRMSNormResidual(nn.Module):
    """Inference-time module: uses the pure-PyTorch path (numerically identical to
    the custom CUDA kernel used during training) since Spaces free tier is CPU-only."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, residual):
        mean_sq = x.float().pow(2).mean(-1, keepdim=True)
        norm = x.float() * torch.rsqrt(mean_sq + self.eps)
        return (residual.float() + self.weight.float() * norm).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out(attn)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, memory, memory_mask=None):
        B, T, D = x.shape
        Bm, S, _ = memory.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(Bm, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(Bm, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class GatedFFN(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        hidden = d_model * mult
        self.w_gate = nn.Linear(d_model, hidden)
        self.w_up = nn.Linear(d_model, hidden)
        self.w_down = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads)
        self.norm1 = FusedRMSNormResidual(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm2 = FusedRMSNormResidual(d_model)
        self.ffn = GatedFFN(d_model)
        self.norm3 = FusedRMSNormResidual(d_model)
        self.pre_ln = nn.LayerNorm(d_model)

    def forward(self, x, memory, memory_mask=None):
        h = self.pre_ln(x)
        x = self.norm1(self.self_attn(h), x)
        x = self.norm2(self.cross_attn(x, memory, memory_mask), x)
        x = self.norm3(self.ffn(x), x)
        return x


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, max_len=96):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, memory, memory_mask=None):
        B, T = tgt_ids.shape
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.tok_emb(tgt_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, memory, memory_mask)
        x = self.final_norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, memory, max_new_tokens=80):
        B = memory.size(0)
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=memory.device)
        for _ in range(max_new_tokens):
            logits = self.forward(ids, memory)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == EOS).all():
                break
        return ids


class AdapterProjection(nn.Module):
    def __init__(self, crnn_hidden=128, d_model=256):
        super().__init__()
        self.text_proj = nn.Linear(crnn_hidden * 2, d_model)
        self.box_mlp = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, word_hidden, boxes_norm):
        tok = self.text_proj(word_hidden) + self.box_mlp(boxes_norm)
        return self.norm(tok)


# ---------------------------------------------------------------------------
# Load checkpoint (safetensors + config.json, from local files or the Hub)
# ---------------------------------------------------------------------------
_crnn, _adapter, _decoder, _cfg = None, None, None, None
_reader = None
_load_error = None


def _resolve_file(filename):
    """Prefer a local copy sitting next to app.py (e.g. bundled into the Space repo);
    otherwise download it from the model repo on the Hub."""
    if os.path.exists(filename):
        return filename
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


def load_models():
    global _crnn, _adapter, _decoder, _cfg, _load_error
    if _crnn is not None or _load_error is not None:
        return
    try:
        weights_path = _resolve_file(WEIGHTS_FILENAME)
        config_path = _resolve_file(CONFIG_FILENAME)

        with open(config_path) as f:
            cfg = json.load(f)

        flat = load_safetensors(weights_path, device=str(DEVICE))
        crnn_sd = {k[len("crnn."):]: v for k, v in flat.items() if k.startswith("crnn.")}
        adapter_sd = {k[len("adapter."):]: v for k, v in flat.items() if k.startswith("adapter.")}
        decoder_sd = {k[len("decoder."):]: v for k, v in flat.items() if k.startswith("decoder.")}

        _crnn = CRNN(img_h=cfg["img_h"], hidden=cfg["crnn_hidden"]).to(DEVICE)
        _crnn.load_state_dict(crnn_sd)
        _adapter = AdapterProjection(cfg["crnn_hidden"], cfg["d_model"]).to(DEVICE)
        _adapter.load_state_dict(adapter_sd)
        _decoder = TinyDecoder(len(cfg["dec_vocab"]), cfg["d_model"], cfg["n_heads"],
                                cfg["n_layers"], cfg["max_len"]).to(DEVICE)
        _decoder.load_state_dict(decoder_sd)
        _crnn.eval(); _adapter.eval(); _decoder.eval()
        globals()["_cfg"] = cfg
    except Exception as e:
        _load_error = str(e)


def get_detector():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _reader


def preprocess_crop(pil_crop, img_h=32, img_w=128):
    im = pil_crop.convert("RGB").resize((img_w, img_h))
    arr = np.array(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def extract_receipt(image: Image.Image):
    load_models()
    if _load_error is not None:
        err = {"error": f"Model checkpoint not found or failed to load: {_load_error}. "
                         f"Make sure '{WEIGHTS_FILENAME}' and '{CONFIG_FILENAME}' are "
                         f"either next to app.py or available at {HF_REPO_ID} on the Hub."}
        return err, [], image, render_field_cards(None)

    reader = get_detector()
    arr = np.array(image.convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=False)
    if not results:
        err = {"error": "No text detected in image."}
        return err, [], image, render_field_cards(None)

    W, H = image.size
    crops, boxes = [], []
    for (poly, text, conf) in results:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        boxes.append(box)
        crops.append(image.crop(box))

    word_imgs = torch.stack([preprocess_crop(c, _cfg["img_h"], _cfg["img_w"]) for c in crops]).to(DEVICE)
    boxes_norm = torch.tensor(
        [[b[0]/W, b[1]/H, b[2]/W, b[3]/H] for b in boxes], dtype=torch.float32).to(DEVICE)

    hidden = _crnn.hidden_states(word_imgs)
    logits = _crnn.fc(hidden)
    pooled = hidden.mean(dim=1)
    memory = _adapter(pooled, boxes_norm).unsqueeze(0)

    texts, confs = [], []
    for i in range(logits.size(0)):
        t, c = ctc_greedy_decode(logits[i].detach().float().cpu())
        texts.append(t); confs.append(c)

    gen_ids = _decoder.generate(memory, max_new_tokens=80)[0].tolist()
    field_values, cur = [], []
    for tid in gen_ids[1:]:
        if tid in (SEP, EOS):
            field_values.append("".join(cur)); cur = []
            if tid == EOS:
                break
        elif tid == PAD:
            continue
        else:
            cur.append(DEC_IDX2TOK.get(tid, ""))
    while len(field_values) < len(FIELDS):
        field_values.append("")
    result = {k: v for k, v in zip(FIELDS, field_values)}

    precontext = [
        {"task": "ocr_word", "text": t, "box": list(b), "confidence": round(c, 3)}
        for t, b, c in zip(texts, boxes, confs)
    ]

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for b, t in zip(boxes, texts):
        draw.rectangle(b, outline="#7C5CFF", width=3)
    return result, precontext, annotated, render_field_cards(result)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_field_cards(result):
    """Renders the 4 extracted fields as a small grid of styled cards instead of
    raw JSON, so the primary result reads like a product UI, not a debug dump."""
    if not result:
        cards = "".join(
            f'<div class="field-card field-card--empty">'
            f'<div class="field-card__icon">{FIELD_ICONS.get(k, "•")}</div>'
            f'<div class="field-card__body">'
            f'<div class="field-card__label">{k}</div>'
            f'<div class="field-card__value field-card__value--empty">—</div>'
            f'</div></div>'
            for k in FIELDS
        )
    else:
        cards = "".join(
            f'<div class="field-card">'
            f'<div class="field-card__icon">{FIELD_ICONS.get(k, "•")}</div>'
            f'<div class="field-card__body">'
            f'<div class="field-card__label">{k}</div>'
            f'<div class="field-card__value">{html.escape(v) if v else "—"}</div>'
            f'</div></div>'
            for k, v in result.items()
        )
    return f'<div class="field-grid">{cards}</div>'


def run(image):
    if image is None:
        return {}, [], None, render_field_cards(None)
    result, precontext, annotated, cards_html = extract_receipt(image)
    return result, precontext, annotated, cards_html


# ---------------------------------------------------------------------------
# Gradio UI — styled to match the Mini-Interfaze pipeline it's fronting
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06)",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #7C5CFF 0%, #5B8DFF 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6B4CEF 0%, #4A7CEF 100%)",
    button_primary_text_color="white",
)

CUSTOM_CSS = """
.app-header {
    background: linear-gradient(135deg, #7C5CFF 0%, #5B8DFF 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 24px -8px rgba(124, 92, 255, 0.45);
}
.app-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.65rem;
    font-weight: 700;
    color: white !important;
}
.app-header p {
    margin: 0;
    opacity: 0.92;
    font-size: 0.95rem;
    color: white !important;
}
.pipeline-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.pipeline-pills span {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.field-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}
.field-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #F7F5FF;
    border: 1px solid #E7E1FF;
    border-radius: 14px;
    padding: 14px 16px;
    transition: box-shadow 0.15s ease;
}
.field-card--empty {
    background: #F8FAFC;
    border: 1px dashed #E2E8F0;
}
.field-card__icon {
    font-size: 1.3rem;
    line-height: 1.4rem;
}
.field-card__label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #7C5CFF;
    margin-bottom: 2px;
}
.field-card__value {
    font-size: 1.0rem;
    font-weight: 600;
    color: #1E1B2E;
    word-break: break-word;
}
.field-card__value--empty {
    color: #94A3B8;
    font-weight: 400;
}
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #64748B;
    margin-bottom: 6px;
}
"""

with gr.Blocks(title="Mini-Interfaze — Receipt Field Extractor", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <div class="app-header">
            <h1>🧾 Mini-Interfaze — Receipt Field Extractor</h1>
            <p>Pretrained text detector → trained CRNN recognizer → adapter projection →
            from-scratch transformer decoder, emitting structured JSON with per-word
            precontext (box + confidence).</p>
            <div class="pipeline-pills">
                <span>CRAFT Detector</span>
                <span>CRNN + BiLSTM + CTC</span>
                <span>Adapter Projection</span>
                <span>Transformer Decoder</span>
                <span>Fused RMSNorm Kernel</span>
                <span>safetensors</span>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown('<div class="section-label">Upload</div>')
            inp = gr.Image(type="pil", label="Receipt photo", height=380)
            btn = gr.Button("✨ Extract Fields", variant="primary", size="lg")
            gr.Markdown(
                "Works best with a clear, upright photo. First run downloads the "
                "pretrained CRAFT detector and the checkpoint, so it may take a moment."
            )

        with gr.Column(scale=6):
            gr.Markdown('<div class="section-label">Extracted Fields</div>')
            field_cards = gr.HTML(render_field_cards(None))

            with gr.Tabs():
                with gr.Tab("🔍 Detected Words"):
                    out_annotated = gr.Image(label=None, show_label=False, height=340)
                with gr.Tab("🧠 Precontext"):
                    out_precontext = gr.JSON(label=None, show_label=False)
                with gr.Tab("{ } Raw JSON"):
                    out_json = gr.JSON(label=None, show_label=False)

    btn.click(run, inputs=inp, outputs=[out_json, out_precontext, out_annotated, field_cards])

if __name__ == "__main__":
    demo.launch()
