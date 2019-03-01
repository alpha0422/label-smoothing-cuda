import torch
import label_smoothing_cuda

class SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=-1):
        loss = label_smoothing_cuda.forward(logits, 1, False)
        ctx.save_for_backward(loss, logits,
            torch.FloatTensor([smoothing]),
            torch.IntTensor([padding_idx]))

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        loss, logits, smoothing, padding_idx = ctx.saved_variables
        grad_logits = label_smoothing_cuda.backward(grad_loss, loss, 1, logits)

        return grad_logits, None, None
