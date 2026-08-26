# Mini-Interfaze: Receipt Field Extractor

A small, from-scratch implementation of the *Interfaze* native-fusion pattern: a
CNN/RNN perceptual encoder (CRNN, OCR) writes directly into the embedding space
a transformer decoder reads, instead of being called as a separate tool. The
decoder emits a fixed-schema JSON object and returns per-field metadata
(`precontext`: bounding boxes + confidence) alongside it.

- **Encoder**: CRNN (CNN + BiLSTM + CTC), trained from scratch on receipt word crops.
- **Adapter**: linear + box-position projection into a shared 256-d embedding space.
- **Decoder**: 4-layer causal transformer, cross-attending over OCR memory tokens,
  with a custom **fused CUDA kernel** (`RMSNorm + residual add`) used in training.
- **Output**: `{"company": ..., "date": ..., "address": ..., "total": ...}` plus a
  `precontext` array of per-word boxes and confidences.

## Files

| File | Purpose |
|---|---|
| `mini_interfaze_receipt_extractor.ipynb` | Full training notebook (Kaggle, GPU T4x2). Trains the model and saves `mini_interfaze_checkpoint.pt`. |
| `app.py` | Standalone Gradio inference app — loads the checkpoint, no training code needed. |
| `requirements.txt` | Dependencies for the app. |
| `mini_interfaze_checkpoint.pt` | Trained weights (produced by the notebook — you generate this, it isn't included here). |

## 1. Train the model

Open `mini_interfaze_receipt_extractor.ipynb` on **Kaggle** with
**Settings → Accelerator → GPU T4 x2**, and run all cells top to bottom. It runs
end-to-end on a built-in synthetic receipt generator with no external dataset
required (swap in real SROIE data by setting `SROIE_ROOT` if you've attached it
as a Kaggle Input).

The last training cell ("Save a checkpoint for deployment") writes
`/kaggle/working/mini_interfaze_checkpoint.pt`. Download it from the **Output**
panel on the right side of the Kaggle notebook page.

## 2. Deploy for free — Hugging Face Spaces

This is the easiest free option: a public URL, no server to manage, CPU inference
is fast enough for this model size.

1. Go to **huggingface.co → New Space**.
2. Choose:
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free) — no GPU needed for inference.
3. In the new Space's file editor (or via git), upload three files:
   - `app.py` (from this folder)
   - `requirements.txt` (from this folder)
   - `mini_interfaze_checkpoint.pt` (the file you downloaded from Kaggle in step 1)
4. The Space will build automatically (installs `requirements.txt`, then runs
   `app.py`). This takes a few minutes the first time (EasyOCR downloads its
   detector weights on first run).
5. You'll get a public URL like `https://huggingface.co/spaces/<you>/mini-interfaze`
   — share it with anyone, no login required to use it.

### Alternative: git push instead of the file editor
```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>
cp /path/to/app.py /path/to/requirements.txt /path/to/mini_interfaze_checkpoint.pt .
git add .
git commit -m "Add Mini-Interfaze receipt extractor"
git push
```

### Alternative free options (if you outgrow Spaces)
- **Streamlit Community Cloud** — similar free tier, if you'd rather write the UI
  in Streamlit than Gradio.
- **Google Colab + `gradio share=True`** — fastest to test, but the link expires
  when the Colab session ends; good for a quick demo, not a persistent app.

## 3. Run locally instead (optional)

```bash
pip install -r requirements.txt
python app.py
```
Opens a local Gradio server (default `http://127.0.0.1:7860`) using the checkpoint
in the same folder.

## Notes

- The custom CUDA kernel (fused RMSNorm+residual) is a **training-time**
  optimization. `app.py` uses an algebraically identical pure-PyTorch fallback for
  inference, so it runs fine on the free CPU tier — no GPU required to serve it.
- Accuracy depends entirely on how much/what data you trained on. The notebook
  defaults to synthetic receipts for a fast, dependency-free demo; for real-world
  use, train on real photographed receipts (e.g. SROIE) for meaningfully useful
  extraction quality.
