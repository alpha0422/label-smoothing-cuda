#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor log_softmax_backward_cuda(
    const at::Tensor &grad,
    const at::Tensor &output,
    int64_t dim,
    const at::Tensor &input);

std::vector<at::Tensor> softmax_xentropy_cuda(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing,
    const bool half_to_float);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> softmax_xentropy_forward(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing,
    const bool half_to_float) {
    CHECK_INPUT(input);
    CHECK_INPUT(labels);

    return softmax_xentropy_cuda(input, labels, smoothing, half_to_float);
}

at::Tensor softmax_xentropy_backward(
    const at::Tensor &grad,
    const at::Tensor &output,
    int64_t dim,
    const at::Tensor &input) {
    CHECK_INPUT(grad);
    CHECK_INPUT(output);
    CHECK_INPUT(input);

    return log_softmax_backward_cuda(grad, output, dim, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_xentropy_forward, "Softmax cross entropy loss with label smoothing forward (CUDA)");
    m.def("backward", &softmax_xentropy_backward, "Softmax cross entropy loss with label smoothing backward (CUDA)");
}
