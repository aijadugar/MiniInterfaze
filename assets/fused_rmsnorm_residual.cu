#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>
#include <cstdint>

// utility
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must a cuda tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CONTIGUOUS(x.is_contiguous(), #x " must a cuda contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x) CHECK_CONTIGUOUS(x)

// Forward CUDA kernel
template <typename scalar_t>
__global__ void fused_rmsnorm_residual_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    float* __restrict__ inv_rms,

    int num_rows,
    int dim,
    float eps
) {
    int row = blockIdx.x;

    if (row >= num_rows) {
        return;
    }

    const scalar_t* x_row = x + static_cast<size_t>(row) * dim;

    const scalar_t* res_row = residual + static_cast<size_t>(row) * dim;

    scalar_t* out_row = out + static_cast<size_t>(row) * dim;


    extern __shared__ float sdata[];

    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {

        float v = static_cast<float>(x_row[i]);

        local_sum += v * v;
    }

    sdata[threadIdx.x] = local_sum;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {

        if (threadIdx.x < stride) {

            sdata[threadIdx.x] +=
                sdata[threadIdx.x + stride];
        }

        __syncthreads();
    }

    float mean_sq = sdata[0] / static_cast<float>(dim);

    float r_inv = rsqrtf(mean_sq + eps);

    if (threadIdx.x == 0) {
        inv_rms[row] = r_inv;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {

        float xv = static_cast<float>(x_row[i]);

        float wv = static_cast<float>(weight[i]);

        float rv = static_cast<float>(res_row[i]);

        float normalized = xv * r_inv;

        float result = rv + wv * normalized;

        out_row[i] = static_cast<scalar_t>(result);
    }
}

// Backward CUDA kernel
template <typename scalar_t>
__global__ void fused_rmsnorm_residual_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ inv_rms,

    scalar_t* __restrict__ grad_x,

    int num_rows,
    int dim
) {
    int row = blockIdx.x;

    if (row >= num_rows) {
        return;
    }

    const scalar_t* go_row = grad_out + static_cast<size_t>(row) * dim;

    const scalar_t* x_row = x + static_cast<size_t>(row) * dim;

    scalar_t* gx_row = grad_x + static_cast<size_t>(row) * dim;

    float r_inv = inv_rms[row];

    extern __shared__ float sdata[];

    float local_dot = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {

        float g = static_cast<float>(go_row[i]) * static_cast<float>(weight[i]);

        float xv = static_cast<float>(x_row[i]);

        local_dot += g * xv;
    }

    sdata[threadIdx.x] = local_dot;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {

        if (threadIdx.x < stride) {

            sdata[threadIdx.x] +=
                sdata[threadIdx.x + stride];
        }

        __syncthreads();
    }

    float dot = sdata[0];

    float r_inv3 = r_inv * r_inv * r_inv;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {

        float g = static_cast<float>(go_row[i]) * static_cast<float>(weight[i]);

        float xv = static_cast<float>(x_row[i]);


        float gx = g * r_inv - xv * dot * r_inv3 / static_cast<float>(dim);

        gx_row[i] = static_cast<scalar_t>(gx);
    }
}

// CUDA launcher: Forward
std::vector<torch::Tensor> fused_rmsnorm_residual_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor residual,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(residual);

    TORCH_CHECK(
        x.dim() >= 1,
        "x must have at least 1 dimension"
    );

    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be 1D"
    );

    TORCH_CHECK(
        x.sizes() == residual.sizes(),
        "x and residual must have the same shape"
    );

    TORCH_CHECK(
        x.size(-1) == weight.size(0),
        "last dimension of x must match weight"
    );

    TORCH_CHECK(
        x.scalar_type() == weight.scalar_type(),
        "x and weight must have the same dtype"
    );

    TORCH_CHECK(
        x.scalar_type() == residual.scalar_type(),
        "x and residual must have the same dtype"
    );

    int dim = static_cast<int>(x.size(-1));

    int64_t num_rows_64 = x.numel() / dim;

    TORCH_CHECK(
        num_rows_64 <= INT_MAX,
        "Too many rows"
    );

    int num_rows = static_cast<int>(num_rows_64);

    auto out = torch::empty_like(x);

    auto inv_rms = torch::empty(
            {num_rows},
            x.options().dtype(torch::kFloat32)
        );

    constexpr int threads = 256;

    int blocks = num_rows;

    size_t shmem = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        x.scalar_type(),
        "fused_rmsnorm_residual_fwd",
        [&] {

            fused_rmsnorm_residual_fwd_kernel<scalar_t>
                <<<blocks, threads, shmem>>>(
                    x.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    inv_rms.data_ptr<float>(),
                    num_rows,
                    dim,
                    static_cast<float>(eps)
                );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, inv_rms};
}

// CUDA launcher: Backward
torch::Tensor fused_rmsnorm_residual_backward_x(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(inv_rms);

    TORCH_CHECK(
        grad_out.sizes() == x.sizes(),
        "grad_out and x must have the same shape"
    );

    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be 1D"
    );

    TORCH_CHECK(
        x.size(-1) == weight.size(0),
        "weight dimension mismatch"
    );

    TORCH_CHECK(
        inv_rms.dim() == 1,
        "inv_rms must be 1D"
    );


    int dim = static_cast<int>(x.size(-1));

    int64_t num_rows_64 = x.numel() / dim;

    TORCH_CHECK(
        num_rows_64 <= INT_MAX,
        "Too many rows"
    );

    int num_rows = static_cast<int>(num_rows_64);


    TORCH_CHECK(
        inv_rms.numel() == num_rows,
        "inv_rms has incorrect number of rows"
    );

    auto grad_x = torch::empty_like(x);

    constexpr int threads = 256;

    int blocks = num_rows;

    size_t shmem = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        x.scalar_type(),
        "fused_rmsnorm_residual_bwd",
        [&] {

            fused_rmsnorm_residual_bwd_kernel<scalar_t>
                <<<blocks, threads, shmem>>>(
                    grad_out.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    inv_rms.data_ptr<float>(),
                    grad_x.data_ptr<scalar_t>(),
                    num_rows,
                    dim
                );
        }
    );


    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_x;
}

// PyTorch bindings
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {
    m.def(
        "fused_rmsnorm_residual_forward",
        &fused_rmsnorm_residual_forward,
        "Fused RMSNorm + Residual Forward"
    );

    m.def(
        "fused_rmsnorm_residual_backward_x",
        &fused_rmsnorm_residual_backward_x,
        "Fused RMSNorm + Residual Backward"
    );
}