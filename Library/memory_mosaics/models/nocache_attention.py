import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import get_device_states, set_device_states

# --------- helper functions -----------

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class Deterministic(nn.Module):
    def __init__(self, net):
        '''Ensures deterministic forward pass inside backward pass under stochasticity.'''
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# ------------------ Attention ------------------

class _AttentionNoCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, activation, mask=None, args=None):
        '''Computes (activation(Q K^T)).V with optional topk sparsity'''
        assert isinstance(activation, Deterministic), 'activation must be wrapped in Deterministic'
        assert mask is None or (mask.dtype == torch.bool and not mask.requires_grad)

        dtype = Q.dtype
        K = K.to(dtype)
        V = V.to(dtype)

        topk = -1
        if args is not None and 'topk' in args and args['topk'] > 0:
            topk = args['topk']
        if args is None:
            args = {}
        args.update({'topk': topk})

        dots = Q.matmul(K.transpose(-1, -2)).to(dtype)
        if mask is not None:
            dots.masked_fill_(mask, max_neg_value(dots))

        top_dots, top_inds = None, None
        if topk > 0:
            mask = None
            top_dots, top_inds = dots.topk(topk, dim=-1, sorted=False)
            attn = dots.zero_().scatter_(
                -1,
                top_inds,
                activation(top_dots, record_rng=activation.training).to(dtype)
            )
        else:
            attn = activation(dots, record_rng=activation.training).to(dtype)

        out = attn.matmul(V)
        ctx.activation = activation
        ctx.args = args
        ctx.save_for_backward(Q, K, V, mask, top_dots, top_inds)
        return out

    @staticmethod
    def backward(ctx, d_out):
        Q, K, V, mask, top_dots, top_inds = ctx.saved_tensors
        args = ctx.args
        activation, topk = ctx.activation, args['topk']
        dtype = Q.dtype

        d_out = d_out.to(dtype)
        V = V.to(dtype)

        d_attn = d_out.matmul(V.transpose(-2, -1))
        matmul_x_t_y = _AttentionNoCache.matmul_x_t_y

        if topk > 0:
            d_top_attn = d_attn.gather(-1, top_inds)
            with torch.enable_grad():
                top_dots.requires_grad = True
                top_attn = activation(top_dots, set_rng=True).to(dtype)
            top_attn.backward(d_top_attn.to(dtype))
            d_top_dots = top_dots.grad
            top_attn = top_attn.detach()
            attn = d_attn.zero_().scatter_(-1, top_inds, top_attn)
            d_V = matmul_x_t_y(attn, d_out)
            d_dots = d_attn.scatter_(-1, top_inds, d_top_dots.to(dtype))
        else:
            dots = Q.matmul(K.transpose(-1, -2)).to(dtype)
            if mask is not None:
                dots.masked_fill_(mask, max_neg_value(dots))
            with torch.enable_grad():
                dots.requires_grad = True
                attn = activation(dots, set_rng=True).to(dtype)
            attn.backward(d_attn)
            d_dots = dots.grad
            d_V = matmul_x_t_y(attn, d_out)

        d_Q = d_dots.matmul(K)
        d_K = matmul_x_t_y(d_dots, Q)
        return d_Q, d_K, d_V, None, None, None

    @staticmethod
    def matmul_x_t_y(x, y):
        dtype = x.dtype
        x = x.to(dtype)
        y = y.to(dtype)
        a, b, c = x.shape[-1], x.shape[-2], y.shape[-1]
        if b*a <= b*c + c*a:
            return x.transpose(-2,-1).matmul(y)
        return y.transpose(-2,-1).matmul(x).transpose(-2,-1)


class AttentionNoCache(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = Deterministic(activation)

    def forward(self, Q, K, V, mask=None, causal_masking=False, args=None):
        assert not causal_masking or mask is None, 'mask should not be provided with causal masking'

        Q_chunks, Lq = 1, Q.shape[-2]
        if args is not None and 'Q_chunk_size' in args:
            Q_chunk_size = args['Q_chunk_size'] if args['Q_chunk_size'] > 0 else Lq
            Q_chunks = max(1, Lq // Q_chunk_size)

        dtype = Q.dtype
        out = torch.zeros(Q.shape[:-1] + (V.shape[-1],), device=Q.device, dtype=dtype)

        for chunk_ids in torch.arange(Lq, device=Q.device).chunk(Q_chunks):
            chunk_mask = None
            if mask is not None:
                assert mask.shape[-2] in [1, Lq]
                chunk_mask = mask if mask.shape[-2] == 1 else mask[..., chunk_ids, :]
            elif causal_masking:
                assert Q.shape[-2] == K.shape[-2]
                chunk_mask = torch.triu(
                    torch.ones(len(chunk_ids), K.shape[-2], device=Q.device, dtype=torch.bool),
                    diagonal=1+chunk_ids[0]
                )
            out[..., chunk_ids, :] = _AttentionNoCache.apply(
                                                            Q[..., chunk_ids, :], K, V, self.activation, chunk_mask, args
                                                            ).to(out.dtype)
        return out
