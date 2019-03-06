import torch
import label_smoothing
import label_smoothing_cuda

import warnings
import random
import numpy as np
import time

def label_smoothing_raw(x, target, padding_idx, smoothing):
    logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)

    non_pad_mask = (target != padding_idx)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)[non_pad_mask]
    smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
    loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
    return loss

def label_smoothing_opt_1(x, target, padding_idx, smoothing):
    logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)

    pad_mask = (target == padding_idx)
    ll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = logprobs.mean(dim=-1)
    loss = (smoothing - 1.0) * ll_loss - smoothing * smooth_loss
    loss.masked_fill_(pad_mask, 0)
    return loss

def validate(ref, val):
    if (ref.norm() - val.norm()) / ref.norm() > 1e-7:
        warnings.warn("Norm difference check failed!")
    else:
        print("Norm difference check passed!")
    
    if torch.equal(ref, val):
        print("Absolute difference check passed!")
    else:
        warnings.warn("Absolute difference check failed!")
    
    rel_diff = torch.abs((ref - val) / ref)
    if torch.ge(rel_diff, 1e-4).any():
        warnings.warn("Relative difference check failed!")
    else:
        print("Relative difference check passed!")

if __name__ == '__main__':
    # Set random seed
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   
    # Set pytorch print precision
    torch.set_printoptions(precision=10)

    # Set label smoothing configuration
    smoothing, padding_idx = 0.1, 0
    #N, T, H = 32, 33, 32320
    N, T, H = 128, 74, 32320
    iters = 1000
    
    # Initialize data
    logits = torch.randn((N*T, H), dtype=torch.half, device='cuda', requires_grad=True)
    labels = torch.randint(0, H, [N*T], device='cuda')
    for i in random.sample(range(N*T), N*T//6):
        labels[i] = padding_idx
    glosses = torch.randn([N*T], dtype=torch.float, device='cuda')
    loss_func = label_smoothing.SoftmaxCrossEntropyLoss.apply
    half_to_float = (logits.dtype == torch.half)

    # Run optimized softmax cross entropy with label smoothing
    losses = loss_func(logits, labels, smoothing, padding_idx, half_to_float)
    glosses = torch.randn_like(losses)
    torch.cuda.synchronize()
    ts = time.time()
    for i in range(iters):
        logits.grad = None
        losses = loss_func(logits, labels, smoothing, padding_idx, half_to_float)
        loss = losses.sum() / N
        loss.backward()
    torch.cuda.synchronize()
    val = logits.grad.clone().detach()
    print("Opt time {:.2f} s elapsed for {} iterations, norm {}".format(
        time.time()-ts, iters, val.norm()))

    # Run original softmax cross entropy with label smoothing
    glosses = glosses[labels != padding_idx]
    torch.cuda.synchronize()
    ts = time.time()
    for i in range(iters):
        logits.grad = None
        losses = label_smoothing_raw(logits, labels, padding_idx, smoothing)
        loss = losses.sum() / N
        loss.backward()
        #losses.backward(glosses)
    torch.cuda.synchronize()
    ref = logits.grad.clone().detach()
    print("Raw time {:.2f} s elapsed for {} iterations, norm {}".format(
        time.time()-ts, iters, ref.norm()))

    # Results validation
    validate(ref, val)

