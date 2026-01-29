# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import math, os
import logging
import torch
import torch.nn as nn
from memory_mosaics.data.dataset_synthetic import SPECIAL
import re
from torch.nn import functional as F
import inspect
from .nocache_attention import AttentionNoCache
try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass  # some poor windows users cant install deepspeed

from .memory import Pmem
import numpy as np
import time

from collections import namedtuple

from .feature_extractor import LeakyAverageCuda
from adasplash import adasplash_no_block_mask as adasplash
#from adasplash import adasplash

class LayerNorm(nn.Module):
    # copied from nanoGPT
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        
class FeatureExtractor(nn.Module):
    def __init__(self, fe_type, shift, norm, kernel_size, n_embd, n_head, bias):
        super().__init__()
        self.fe_type = fe_type
        self.norm = norm # without trainable parameter
        self.n_head = n_head
        self.n_embd = n_embd
        if self.fe_type == 'linearconv':# or self.fe_type =='reluconv':
            self.shiftpad = nn.ZeroPad1d((kernel_size-1 - shift, shift))
            self.conv = nn.Conv1d(in_channels=n_embd, out_channels= n_embd, kernel_size=kernel_size, bias=bias)

        elif self.fe_type == 'lowrlinearconv':
            assert kernel_size == 2, 'fe_type == lowrlinearconv only support kernel_size=2'
            self.shiftpad = nn.ZeroPad1d((-shift, shift))
            self.linear = nn.Linear(n_embd, n_embd, bias=bias)
            self.coef = nn.Parameter(torch.ones(n_head).uniform_(0,1))
        else:
            raise NotImplementedError      
        
    def forward(self, x):
        
        B,T,C= x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        nh = self.n_head
        hs = self.n_embd // nh

        if  self.fe_type == 'lowrlinearconv': # x.shape  (B, T, C)
            x = self.linear(x).transpose(1,2).view(B, C, T) # B,C, T
            y = self.shiftpad(x).view(B, nh, hs, T).transpose(2,3) * (1- self.coef.view(1,nh,1,1)) + \
                                x.view(B,nh,hs,T).transpose(2,3) * (self.coef.view(1,nh,1,1))
        elif self.fe_type == 'linearconv':
            x = x.transpose(1,2) # (B, C, T)
            y = self.conv(self.shiftpad(x)).view(B, nh, hs, T).transpose(2,3) # (B, nh, T, hs)
        else:
            raise NotImplementedError

        if self.norm:
            y = y / (y.norm(dim=-1, keepdim=True)+ 1e-6)

        return y 

class LeakyAvg(nn.Module):
    def __init__(self, block_size, n_head):
        super().__init__()
        coef = torch.zeros(block_size, block_size)
        for i in range(block_size):
            coef = torch.diagonal_scatter(coef, -torch.ones(block_size-i)*i, -i)
        self.register_buffer('coef', coef)
        self.exp_scaling = 10
        self.leaky_key_beta = nn.Parameter(torch.linspace(0.5, 5, n_head).view(1, n_head, 1, 1)/self.exp_scaling)

    def forward(self, k):
        B, nh, T, hs = k.size()
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_scaling
        coef = self.coef[:T,:T].view(1,1,T,T)

        coef_exp_arg = (coef * leaky_key_beta)
        coef = torch.exp(coef_exp_arg)
        return coef.tril() @ k

class InContextAssoMemBlock(nn.Module): # leakyaverage key, then normalize. convolution on value, then normalize. cheat first token
    
    def __init__(self, method = "softmax", task=None, n_head=12, n_embd=768, v_shift=1, block_size=512, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, config={}):
        super().__init__()
        #assert n_embd % n_head == 0
        self.config = config
        self.n_head = n_head
        self.n_embd = n_embd
        self.v_shift = v_shift 
        self.block_size = block_size 
        self.bias = bias
        self.k_kernel_size = k_kernel_size
        self.v_kernel_size = v_kernel_size
        self.method = method
        self.task = task
        self.att_shift = 0
        self.value_beta_init = -0.5 
        self.exp_param_scale = 10
       
        
        self.causal_mask = torch.tril(torch.ones(self.block_size+self.v_shift+self.att_shift, self.block_size+self.v_shift+self.att_shift))[:self.block_size, self.v_shift+self.att_shift:].view(1,1,self.block_size,self.block_size) != 0
        
        
        self.v_featurizer = FeatureExtractor(fe_type=v_fe_type, shift=self.v_shift, norm=True,  \
                                            kernel_size=v_kernel_size, n_embd=n_embd, n_head = n_head, bias = bias)

        self.ic_dropout = nn.Dropout(ic_dropout)

        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(hd_dropout)
        if "adapt" in self.method:
            self.kernel_beta0 = nn.Parameter(
                torch.ones(self.n_head) * (1.5 / self.exp_param_scale)
            )

            # β1 > 0
            self.kernel_beta1 = nn.Parameter(
                torch.ones(self.n_head) * (1.5 / self.exp_param_scale)
            )

            # 0 < α < 1
            self.kernel_alpha = nn.Parameter(
                torch.ones(self.n_head) * (1/3)
            )
        else:
            self.kernel_beta = nn.Parameter(torch.ones(self.n_head)/self.exp_param_scale) # for fp16, max float is 65536. kernel_beta < math.log(math.sqrt(65536)) ~5.5
        self.kernel_beta_upbound = 5 # assume fp16. 

        self.value_beta  = nn.Parameter(torch.ones(self.n_head) * (self.value_beta_init)/self.exp_param_scale)
        if self.task == "reverse" or self.task =="sort" or self.task=="mqmtar":
            block_size = 4096 # A big number to not get errors
        self.leaky_average = LeakyAverageCuda(n_embd=self.n_embd, n_head=self.n_head, max_seq_length=block_size, leaky_per_head=True, sep_w_on_t=False, linear=True, bias=bias)
        #self.leaky_average = LeakyAvg( n_head=self.n_head, block_size=block_size)


    def forward(self, x, attn_mask=None, ctx=None):
        B,T,C= x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        hs = self.n_embd // self.n_head
        nh = self.n_head
        
    # clone to avoid autograd saved-tensor versioning issues when using compiled graphs
        k = self.leaky_average(x).view(B,T, nh,hs).transpose(1,2)
        raw_k = k / (k.norm(dim=-1, keepdim=True)+ 1e-6)
        beta = (torch.exp(torch.clamp(self.kernel_beta*self.exp_param_scale, max=self.kernel_beta_upbound))).view(1,nh,1,1)  #(B, nh, T, hs)
        k = raw_k * beta.expand(B, nh, T,hs)  #(B, nh, T, hs)
        v = self.v_featurizer(x) *  ((torch.exp(torch.clamp(self.value_beta*self.exp_param_scale, max=self.kernel_beta_upbound))).view(1,nh,1,1).expand(B, nh, T,hs))
        self.block_size = T
        self.causal_mask = torch.tril(torch.ones(self.block_size+self.v_shift+self.att_shift, self.block_size+self.v_shift+self.att_shift))[:self.block_size, self.v_shift+self.att_shift:].view(1,1,self.block_size,self.block_size) != 0
        self.causal_mask = self.causal_mask.to(k.device)
        if self.task is not None:
            self.causal_mask = self.causal_mask & attn_mask   # broadcast on dim 0
        
            ##########################
        y = torch.zeros_like(v)

        if self.method == 'softmax':
            y[:, :, self.v_shift+self.att_shift:] = F.scaled_dot_product_attention(
                k[:, :, self.v_shift+self.att_shift:], 
                k, 
                v, 
                attn_mask=self.causal_mask[:, :, self.v_shift+self.att_shift:T, :T].to(k.device),
                scale=1.0,
                is_causal=False
            )

        elif "top" in self.method and "softmax" in self.method:
            f = nn.Softmax(-1)
            match = re.search(r'top(\d+)_', self.method)
            top_k = match.group(1)
            activation = lambda x: f(x).to(x.dtype) 
            args = {'topk':int(top_k), 'Q_chunk_size':512}
            q = k[:, :, self.v_shift + self.att_shift:, :]  # (B, H, T_shifted, D)
            # add dummy last row
            dummy = torch.zeros_like(q[:, :, :1, :])       # (B, H, 1, D)
            q = torch.cat([q, dummy], dim=2) 
            o = AttentionNoCache(activation)(q, k, v, args=args, mask=None, causal_masking=True) 
            # discard last dummy row
            o = o[:, :, :-1, :]  # (B, H, T_shifted, D)

            # avoid in-place write
            prefix = y[:, :, :self.v_shift+self.att_shift]
            y = torch.cat([prefix, o], dim=2)
        
        elif "uniform" in self.method or "relu" in self.method:

            # make contiguous clones to avoid views/aliasing that can confuse autograd when
            # running with TorchDynamo / compiled graphs
            V = v.contiguous().clone()

            if "uniform" in self.method:
                q = k[:, :, self.v_shift + self.att_shift:, :].contiguous()
                K = k.contiguous()

                # (B, H, Q, K)
                scores = torch.matmul(q, K.transpose(-1, -2))

                # apply causal mask BEFORE topk
                scores = scores.masked_fill(
                    ~self.causal_mask[:, :, :scores.size(-2), :scores.size(-1)],
                    float('-inf')
                )

                match = re.search(r'top(\d+)', self.method)
                top_k_val = int(match.group(1))
                top_k = min(top_k_val, K.size(2))
                
                # 1. valid causal keys
                valid = torch.isfinite(scores)                 # bool
                valid_count = valid.sum(dim=-1, keepdim=True)  # (B,H,Q,1)

                # 2. handle rows with zero valid keys
                has_any = valid_count > 0

                # 3. safe top-k (still returns k indices)
                k = top_k
                top_idx = torch.topk(scores, k, dim=-1).indices

                # 4. binary selection mask
                mask = torch.zeros_like(scores)
                mask.scatter_(-1, top_idx, 1.0)

                # 5. remove accidentally selected -inf positions
                mask = mask * valid.float()

                # 6. normalize by effective support size
                denom = mask.sum(dim=-1, keepdim=True)
                mask = torch.where(has_any, mask / denom.clamp_min(1), mask)
                y_out = mask.float() @ V   # (B, H, Q, D)

            elif self.method == "norm_relu":
                q = raw_k[:, :, self.v_shift + self.att_shift:, :].contiguous()
                K = raw_k.contiguous()
                beta = torch.exp(
                    torch.clamp(
                        self.kernel_beta * self.exp_param_scale,
                        max=self.kernel_beta_upbound
                    )
                ).view(1, nh, 1, 1)   # <-- STOP HERE, NO EXPAND

                h = beta.reciprocal()         # safer than 1 / beta
                gamma = h.square().mul_(0.5) # in-place
                b = 1.0 - 2.0 * beta.square()  # since 1 - 2 / h^2 = 1 - 2 * beta^2

                scores = torch.matmul(q, K.transpose(-1, -2))

                mask = self.causal_mask[:, :, :scores.size(-2), :scores.size(-1)]
                
                # 1) Zero out masked positions BEFORE ReLU
                scores = scores.masked_fill(~mask, 0.0)

                # 2) ReLU
                scores = F.relu(scores)

                # 3) Row sum over valid keys
                row_sum = scores.sum(dim=-1, keepdim=True)   # (B,H,Q,1)

                # 4) Uniform over valid positions only
                valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
                uniform = mask.to(scores.dtype) / valid_count

                # 5) Normalize or fallback
                weights = torch.where(
                    row_sum > 0,
                    scores / row_sum.clamp_min(1e-6),   # normal normalized ReLU
                    uniform             # indeterminate → uniform over valid
                )
                y_out = torch.matmul(weights, V)
            
            elif self.method == "relumax":
                q = raw_k[:, :, self.v_shift + self.att_shift:, :].contiguous()
                K = raw_k.contiguous()
                beta = torch.exp(
                    torch.clamp(
                        self.kernel_beta * self.exp_param_scale,
                        max=self.kernel_beta_upbound
                    )
                ).view(1, nh, 1, 1)   # <-- STOP HERE, NO EXPAND

                h = beta.reciprocal()        # bandwidth
                gamma = h.square()          # gamma = h^2
                b = 1.0                     # fixed margin

                scores = torch.matmul(q, K.transpose(-1, -2))   # (B, nh, Tq, Tk)

                mask = self.causal_mask[:, :, :scores.size(-2), :scores.size(-1)]

                # push masked entries far negative BEFORE max
                scores = scores.masked_fill(~mask, -1e12)

                # center
                scores = scores - scores.max(dim=-1, keepdim=True).values

                # scale
                scores = scores / gamma
                mask01 = mask.to(dtype=scores.dtype).expand_as(scores)

                scores = scores + b * mask01
                # relu
                scores = F.relu(scores)

                # compute row sum
                row_sum = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                # safe division: if sum==0, output 0
                weights = torch.where(row_sum > 0, scores / row_sum, torch.zeros_like(scores))
                y_out = torch.matmul(weights, V)

            # avoid in-place write into y
            prefix = y[:, :, :self.v_shift+self.att_shift]
            y = torch.cat([prefix, y_out], dim=2)
        else:
            # AdaSplash / kernel attention
            q = k[:, :, self.v_shift + self.att_shift:, :]  # (B, H, T_shifted, D)
            # add dummy last row
            dummy = torch.zeros_like(q[:, :, :1, :])       # (B, H, 1, D)
            q = torch.cat([q, dummy], dim=2)               # (B, H, T_shifted+1, D)

            if self.method == "epanechnikov":
                alpha = 2.0
            elif self.method == "biweight": 
                alpha = 1.5
            elif self.method == "triweight":
                alpha = 4/3

            q = q.contiguous()*(q.size(-1)**0.5)  # scale q to match softmax scale 1   
            k = k.contiguous()
            v = v.contiguous()
            o = adasplash(
                q, k, v,
                alpha=alpha,
                niter=5,
                is_causal=True,
                varlen=ctx
            )

            # discard last dummy row
            o = o[:, :, :-1, :]  # (B, H, T_shifted, D)

            # avoid in-place write
            prefix = y[:, :, :self.v_shift+self.att_shift]
            y = torch.cat([prefix, o], dim=2)

        ##########################
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

    
        return y

class PersistentAssoMemBlock(nn.Module): # leakyaverage key, then normalize. convolution on value, then normalize. cheat first token
    
    def __init__(self, task="mqmtar", n_head=12, n_embd=768, block_size=512, \
                k_fe_type='linearconv',\
                k_kernel_size=1,\
                ic_dropout=0.05, hd_dropout=0.05, \
                bias=False, \
                pmem_count=1, pmem_size=2688, \
                config={}):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.config = config
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size 
        self.bias = bias
        self.task = task
        self.att_shift = 0
        self.value_beta_init = -0.5 
        self.exp_param_scale = 10
       
        
        self.pmem = Pmem(pmem_count, pmem_size, self.n_embd//self.n_head, self.n_head, ic_dropout)
        if self.task == "reverse" or self.task =="sort" or self.task=="mqmtar":
            block_size = 4096 # A big number to not get errors
        # Wrap with SafeLeakyAverage to avoid capturing the custom CUDA op in torch._dynamo graphs
        self.leaky_average = LeakyAverageCuda(n_embd=self.n_embd, n_head=self.n_head, max_seq_length=block_size, leaky_per_head=True, sep_w_on_t=False, linear=True, bias=bias)
        #self.leaky_average = LeakyAvg( n_head=self.n_head, block_size=block_size)


        self.ic_dropout = nn.Dropout(ic_dropout)
        
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(hd_dropout)
            
       
        #self.leaky_key_beta = nn.Parameter( torch.linspace(0.5, 5, self.n_head) / self.exp_param_scale) 
        self.kernel_beta = nn.Parameter(torch.ones(self.n_head)/self.exp_param_scale) # for fp16, max float is 65536. kernel_beta < math.log(math.sqrt(65536)) ~5.5
        self.kernel_beta_upbound = 5 # assume fp16. 

        self.value_beta  = nn.Parameter(torch.ones(self.n_head) * (self.value_beta_init) / self.exp_param_scale)
    
    @torch._dynamo.disable
    def leaky_average_eager(self, x):
        return self.leaky_average(x)

    def forward(self, x):
        #print(x.dtype)
        B,T,C= x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        hs = self.n_embd // self.n_head
        nh = self.n_head
        # clone to avoid autograd saved-tensor versioning issues when using compiled graphs
        k = self.leaky_average_eager(x) \
                .view(B, T, nh, hs).transpose(1, 2)
        
        k = k / k.norm(dim=-1, keepdim=True)
        k = k *  (torch.exp(torch.clamp(self.kernel_beta*self.exp_param_scale, max=self.kernel_beta_upbound))**2 ).view(1,nh,1,1).expand(B, nh, T,hs)  #(B, nh, T, hs)
        # k times exp(kernel_beta)**2 because one scalar is for persistent memory 
        
        y = self.pmem(k) * (torch.clamp(torch.exp(self.value_beta*self.exp_param_scale).view(1,nh,1,1), max=self.kernel_beta_upbound).expand(B, nh, T,hs))
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))


        return y

class AttnOnlyBlock(nn.Module):
    def __init__(self, method, task=None, n_head=12, n_embd=768, v_shift=1, block_size=512, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, 
                pmem_count=1, pmem_size=2688, pre_ln=True,attn_only=False, \
                config={}):
        super().__init__()
        self.attn_only = attn_only

        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.attn = InContextAssoMemBlock(method=method, task=task, n_head=n_head, n_embd=n_embd, v_shift=v_shift, block_size=block_size, \
                                            k_fe_type=k_fe_type, v_fe_type=v_fe_type, \
                                            k_kernel_size=k_kernel_size, v_kernel_size=v_kernel_size, \
                                            ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                                            bias=bias, \
                                            config=config)
        if not self.attn_only:
            self.ln2 = LayerNorm(n_embd, bias=bias)
            self.pmem = PersistentAssoMemBlock(n_head=n_head, task=task, n_embd=n_embd, block_size=block_size, \
                                                k_fe_type=k_fe_type, 
                                                k_kernel_size=k_kernel_size, 
                                                ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                                                bias=bias, \
                                                pmem_count=pmem_count, pmem_size = pmem_size,\
                                                config=config)
            
        self.pre_ln = pre_ln

    def forward(self, x, attn_mask=None, ctx=None):

        if self.pre_ln: # pre layernorm
            x = x + self.attn(self.ln1(x), attn_mask=attn_mask, ctx=ctx)
            if not self.attn_only:
                x = x + self.pmem(self.ln2(x))
        else:           # post layernorm
            x = self.ln1(x)
            x = x + self.attn(x, attn_mask=attn_mask, ctx=ctx)
            if not self.attn_only:
                x = self.ln2(x)
                x = x + self.pmem(x)
        return x

class StackAssoMem(nn.Module):
    def __init__(self, method, vocab_size, n_head=12, n_embd=768, n_layer=12, v_shift=1, block_size=512, task=None, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, 
                pmem_count=1, pmem_size=2688, pre_ln=True, weight_tying=True, skip_tokens=0, attn_only=False, \
                config={}):
        super().__init__()

        self.config = config
        self.weight_tying = weight_tying 
        self.eval_status = []
        self.vocab_size = vocab_size
        self.skip_tokens = skip_tokens
        self.n_embd = n_embd
        self.attn_only = attn_only
        self.task = task
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, self.n_embd)
        self.blocks = nn.ModuleList(
            [AttnOnlyBlock(method=method, task=task, n_head=n_head, n_embd=n_embd, v_shift=v_shift,
                        block_size=block_size, k_fe_type=k_fe_type, v_fe_type=v_fe_type,
                        k_kernel_size=k_kernel_size, v_kernel_size=v_kernel_size,
                        ic_dropout=ic_dropout, hd_dropout=hd_dropout,
                        bias=bias, pmem_count=pmem_count, pmem_size=pmem_size,
                        pre_ln=pre_ln, attn_only=attn_only, config=config)
            for _ in range(n_layer)]
        )
        self.ln_out = LayerNorm(self.n_embd, bias=bias)
        self.head = nn.Linear(self.n_embd, vocab_size, bias=False)
        if self.weight_tying:
            print('weight_tying')
            self.emb.weight = self.head.weight        

        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                fanin = p.shape[1]
                torch.nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(fanin) / math.sqrt(n_layer))

        self.device = "cpu"
        
        
    def forward(self, idx, targets=None, logits_only = False, ctx=None): # state=None,return_hidden_outputs=None):#, futures=None, storyid=None, skip_next=False, skip_future=False):  # fugures (b,n,t)
        device = idx.device
        b, t = idx.size()
        attn_mask = None
        if self.task is not None:
            PAD = SPECIAL[self.task]["<pad>"]
            B, T = idx.size()
            device = idx.device

            # 1. Causal mask [1,1,T,T]
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)   # [1,1,T,T]

            # 2. Padding mask [B,1,1,T]
            pad_mask = (idx != PAD).unsqueeze(1).unsqueeze(2)   # [B,1,1,T]

            # 3. Combine → [B,1,T,T]
            attn_mask = causal_mask & pad_mask   # broadcast on dim 0
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, ctx=ctx)
        x = self.ln_out(x)

        #logits, loss =  None, 0
        logits_scale = 1.0 / math.sqrt(self.n_embd)

        if logits_only:
            logits = logits_scale * self.head(x[:,self.config.skip_tokens:])
            return logits

       
        if targets is not None: 
            logits = logits_scale * self.head(x[:,self.config.skip_tokens:])
            if self.task is not None:
                device = idx.device
                sep_id = SPECIAL[self.task]["<sep>"]
                sep_positions = (idx == sep_id).int()  # 1 where SEP
                # index of last SEP in each row (if multiple, take first)
                sep_idx = sep_positions.argmax(dim=1)  # [B], int

                # 2. Build mask: positions > sep_idx and not PAD
                positions = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
                loss_mask = (positions >= sep_idx.unsqueeze(1)) & (idx != PAD)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets[:,self.config.skip_tokens:].contiguous().view(-1), ignore_index=-1, reduction='none')
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.view(-1).sum()
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets[:,self.config.skip_tokens:].contiguous().view(-1), ignore_index=-1)

            with torch.no_grad():
               self.eval_status = [loss.detach().item()]

       
        if targets is None:# and futures is None:
            logits = logits_scale * self.head(x[:, [-1], :]) #if not skip_next else None 
            loss = None


        return logits, loss



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fanin = module.weight.shape[1]
            # print(fanin)
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            assert len(module.weight.shape) == 3
            #(Cout, Cin, kernelsize)
            fanin = module.weight.shape[1] * module.weight.shape[2]
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1)
    

    def to(self, device, **kwargs):
        super().to(device, **kwargs)

        for i, block in enumerate(self.blocks):
            if i==0:
                attn_mask = block.attn.causal_mask.to(device, **kwargs)
                #leakyavg_coef =   block.attn.leakyavg_coef.to(device, **kwargs)
                #leaky_attn_mask = block.attn.leaky_attn_mask.to(device, **kwargs)
           
            #block.attn.leaky_attn_mask = leaky_attn_mask
            block.attn.causal_mask = attn_mask
            #block.attn.leakyavg_coef = leakyavg_coef
            # if not self.attn_only:
            #     block.pmem.leakyavg_coef = leakyavg_coef
            #     #block.pmem.leaky_attn_mask = leaky_attn_mask
                
            block.attn.device=device

        #self.share_mask_if_there_is_a_mask()
        self.device = device



    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # TODO l2 weight decay applied on memory but not time mix correficient

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if "time_" in n   or 'gate' in n or p.dim() < 2:
                nodecay_params.append(p)
                #if int(os.environ["RANK"]) <=0:
                #print("nodecay params", n)
            else:
                decay_params.append(p)
                #if int(os.environ["RANK"]) <=0:
                #print("decay params", n)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        # print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    def checkparams(self):
        return {}


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, storyid=None, temperature=1.0, top_k=None, stop_token=None, skip_future=True, future_topk=2):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx_future = [[]]*idx.size(0)

        #logits_future_mean = None #logits_future.mean(axis=1)
                
        for _ in range(max_new_tokens):
            # try:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.block_size
                else idx[:, -self.block_size :]
            )


            # By default, logits + softmax + crossentropy, 
            # logits_future + sigmoid + binarycrossentropy
            logits, _ = self(idx_cond)#, storyid=storyid, skip_future=skip_future)
            
            

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities

            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # stop token. it is time to stop.
            if stop_token is not None and idx_next == stop_token:
                break 

        
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)


        #idx_future = None if len(idx_future) == 0 else idx_future
        return idx, None



    @torch.no_grad()
    def beam_generate(self, idx, max_new_tokens, beam_width, temperature=1.0, stop_token=None):
        bank = []
        assert idx.size(0)==1
        
        idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
        
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        topkprob, topkidx = torch.topk(probs, k=beam_width, axis=1)
        #print(topkprob, topkidx)
        for i, var in enumerate(topkidx[0]):
            if stop_token is not None and var == stop_token:
                continue
            bank.append([torch.cat([idx,var.view(1,1)], dim=1), topkprob[0][i].item(), 1])
            

        final_bank = []
        while len(bank) > 0:
            temp_bank = []
            for (idx, prob, length) in bank:

                if length >= max_new_tokens:
                    final_bank.append([idx, prob, length])
                    continue

                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)

                topkprob, topkidx = torch.topk(probs, k=beam_width, axis=1)
                    
                for i, var in enumerate(topkidx[0]):
                    if var == stop_token:
                        final_bank.append([idx, prob, length])
                        continue
                    temp_bank.append([torch.cat([idx,var.view(1,1)], dim=1), topkprob[0][i].item()*prob, length+1])
            
            if len(temp_bank) == 0:
                break
            bank = temp_bank
            topkidx = np.argsort([var[1] for var in bank])[-beam_width:]
            #print(topkidx)
            bank = [bank[var] for var in topkidx]

        
        idx = final_bank[np.argmax([var[1] for var in bank])][0]
        return idx, None 

