#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor log_softmax_cuda(
    const at::Tensor &input,
    const int64_t dim,
    const bool half_to_float);

at::Tensor log_softmax_backward_cuda(
    const at::Tensor &grad,
    const at::Tensor &output,
    int64_t dim,
    const at::Tensor &input);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor label_smoothing_forward_cuda(
    at::Tensor &input,
    const int64_t dim,
    const bool half_to_float){
    CHECK_INPUT(input);

    return log_softmax_cuda(input, dim, half_to_float);
}

at::Tensor label_smoothing_backward_cuda(
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
    m.def("forward", &label_smoothing_forward_cuda, "Label smoothing forward (CUDA)");
    m.def("backward", &label_smoothing_backward_cuda, "Label smoothing backward (CUDA)");
}
