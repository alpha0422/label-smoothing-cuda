import torch
import label_smoothing_cuda

class SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=0, half_to_float=False):
        logprobs, losses = label_smoothing_cuda.forward(
            logits, labels, smoothing, half_to_float)
        losses.masked_fill_(labels==padding_idx, 0)

        ctx.save_for_backward(logprobs, labels,
            torch.FloatTensor([smoothing]),
            torch.LongTensor([padding_idx]),
            torch.ByteTensor([half_to_float]))

        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logprobs, labels, smoothing, padding_idx, half_to_float = ctx.saved_variables
        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels==padding_idx.item(), 0)
        grad_logits = label_smoothing_cuda.backward(
            grad_loss.contiguous(), logprobs, labels,
            smoothing.item(), half_to_float.item())

        return grad_logits, None, None, None, None
