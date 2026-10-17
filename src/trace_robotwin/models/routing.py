import torch




class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # Solve sum relu(logits/2 - threshold)^2 = 1 on the last axis.
        x=(logits-logits.amax(-1,keepdim=True))/2
        lo=x.amax(-1,keepdim=True)-1;hi=x.amax(-1,keepdim=True)
        for _ in range(64):
            mid=(lo+hi)/2
            mass=(x-mid).clamp_min(0).square().sum(-1,keepdim=True)
            lo=torch.where(mass>1,mid,lo);hi=torch.where(mass>1,hi,mid)
        p=(x-(lo+hi)/2).clamp_min(0).square()
        ctx.save_for_backward(p)
        return p
    @staticmethod
    def backward(ctx, grad):
        p,=ctx.saved_tensors;s=p.sqrt()
        v=(grad*s).sum(-1,keepdim=True)/s.sum(-1,keepdim=True)
        return s*(grad-v)

def entmax15(logits):
    # Preserve double for gradcheck; use FP32 for BF16/FP16 production routing.
    return _Entmax15.apply(logits if logits.dtype==torch.float64 else logits.float())

def sparse_route(logits, step, start=3000, full=6000):
    eta=min(1.,max(0.,(step-start)/(full-start)))
    soft=logits.float().softmax(-1);sparse=entmax15(logits)
    return (1-eta)*soft+eta*sparse,soft,eta

