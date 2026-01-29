# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import math, os
import logging
from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect

import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import re
try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass  # some poor windows users cant install deepspeed

from .memory import Pmem
import numpy as np
import time

from collections import namedtuple


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
            y = y / y.norm(dim=-1, keepdim=True)

        return y 
 
class InContextAssoMemBlock(nn.Module): # leakyaverage key, then normalize. convolution on value, then normalize. cheat first token
    
    def __init__(self, method="softmax", n_head=12, n_embd=768, v_shift=1, block_size=512, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, config={}):
        super().__init__()
        assert n_embd % n_head == 0
        self.config = config
        self.n_head = n_head
        self.n_embd = n_embd
        self.method = method
        self.v_shift = v_shift 
        self.block_size = block_size 
        self.bias = bias
        self.k_kernel_size = k_kernel_size
        self.v_kernel_size = v_kernel_size


        self.att_shift = 0
        self.value_beta_init = -0.5 
        self.exp_param_scale = 10
       
        
        self.attn_mask = torch.tril(torch.ones(self.block_size+self.v_shift+self.att_shift, self.block_size+self.v_shift+self.att_shift))[:self.block_size, self.v_shift+self.att_shift:].view(1,1,self.block_size,self.block_size) == 0
        self.leaky_attn_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1,1,self.block_size,self.block_size) == 0
    
            
        self.k_featurizer = FeatureExtractor(fe_type=k_fe_type, shift=0,            norm=False, \
                                            kernel_size=k_kernel_size, n_embd=n_embd, n_head = n_head, bias = bias)
        self.v_featurizer = FeatureExtractor(fe_type=v_fe_type, shift=self.v_shift, norm=True,  \
                                            kernel_size=v_kernel_size, n_embd=n_embd, n_head = n_head, bias = bias)

        self.ic_dropout = nn.Dropout(ic_dropout)

        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(hd_dropout)
            
        # # leakyavg coef 
        self.leakyavg_coef = torch.zeros(self.block_size, self.block_size) 
  
        for i in range(self.block_size):
            self.leakyavg_coef = torch.diagonal_scatter(self.leakyavg_coef, -torch.ones(self.block_size-i)*i, -i)

        
        self.leaky_key_beta = nn.Parameter( torch.linspace(0.5, 5, self.n_head)/self.exp_param_scale)        
        self.kernel_beta = nn.Parameter(torch.ones(self.n_head)/self.exp_param_scale) # for fp16, max float is 65536. kernel_beta < math.log(math.sqrt(65536)) ~5.5
        self.kernel_beta_upbound = 5 # assume fp16. 

        self.value_beta  = nn.Parameter(torch.ones(self.n_head) * (self.value_beta_init)/self.exp_param_scale)
        
      

    def forward(self, x):

        B,T,C= x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        hs = self.n_embd // self.n_head
        nh = self.n_head

        k = self.k_featurizer(x)
        leaky_key_beta = self.leaky_key_beta.abs()*self.exp_param_scale
        leakyavg_coef = torch.exp(self.leakyavg_coef[:T,:T].view(1,1,T,T).expand(1, nh, T, T) \
                                    * leaky_key_beta.view(1,nh,1,1).expand(1,nh,T,T)) * \
                                    (1 - torch.exp(-leaky_key_beta)).view(1,nh,1,1).expand(1,nh, T, T)
        leakyavg_coef = leakyavg_coef.masked_fill(self.leaky_attn_mask[:,:,:T,:T], 0)
        k = leakyavg_coef.expand(B, nh, T, T) @ k 

        k = k / k.norm(dim=-1, keepdim=True)
        raw_k = k
        k = k *  (torch.exp(torch.clamp(self.kernel_beta*self.exp_param_scale, max=self.kernel_beta_upbound))).view(1,nh,1,1).expand(B, nh, T,hs)  #(B, nh, T, hs)
        
        
        v = self.v_featurizer(x) *  ((torch.exp(self.value_beta*self.exp_param_scale)).view(1,nh,1,1).expand(B, nh, T,hs))
        
        
        att = k @ k.transpose(-1,-2) #(k / math.sqrt(hs)) @ (k.transpose(-1,-2) / math.sqrt(hs)) #math.sqrt(hs) # (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)  
        valid = ~self.attn_mask[:, :, :T, :T]  # True where positions are allowed  
        selected = att * valid   
        raw=att
        att = att.masked_fill(self.attn_mask[:,:,:T,:T], float('-inf'))                   # zero out masked positions
        att_slice = att[:,:,self.v_shift+self.att_shift:]
        selected = selected[:,:,self.v_shift+self.att_shift:]
        if self.method=="softmax":
            att[:,:,self.v_shift+self.att_shift:] = F.softmax(att[:,:,self.v_shift+self.att_shift:], dim=-1) # for position [self.v_shift+self.att_shift:], normal attention.
        elif self.method == "epanechnikov":
            att[:,:,self.v_shift+self.att_shift:] = sparsemax(att[:,:,self.v_shift+self.att_shift:], dim=-1) # for position [self.v_shift+self.att_shift:], normal attention.
        elif self.method == "triweight":
            att[:,:,self.v_shift+self.att_shift:] = entmax_bisect(att[:,:,self.v_shift+self.att_shift:], dim=-1, alpha=4/3) 
        elif self.method == "biweight":
            att[:,:,self.v_shift+self.att_shift:] = entmax_bisect(att[:,:,self.v_shift+self.att_shift:], dim=-1, alpha=1.5) 
        elif "top" in self.method and "softmax" in self.method:
            match = re.search(r'top(\d+)_', self.method)
            top_k = match.group(1)
            k_val = int(top_k)  # or make it a parameter
            # get top-k indices
            topk_vals, topk_idx = att_slice.topk(k_val, dim=-1)
            # create a mask of everything else
            mask = torch.full_like(att_slice, float('-inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            # softmax over the top-k values
            att[:,:,self.v_shift+self.att_shift:] = F.softmax(mask, dim=-1)
        
        elif "uniform" in self.method:
            match = re.search(r'top(\d+)', self.method)
            top_k = int(match.group(1))

            
            # 1. valid causal keys
            valid = torch.isfinite(att_slice)                 # bool
            valid_count = valid.sum(dim=-1, keepdim=True)  # (B,H,Q,1)

            # 2. handle rows with zero valid keys
            has_any = valid_count > 0

            # 3. safe top-k (still returns k indices)
            k = top_k
            top_idx = torch.topk(att_slice, k, dim=-1).indices

            # 4. binary selection mask
            mask = torch.zeros_like(att_slice)
            mask.scatter_(-1, top_idx, 1.0)

            # 5. remove accidentally selected -inf positions
            mask = mask * valid.float()

            # 6. normalize by effective support size
            denom = mask.sum(dim=-1, keepdim=True)
            mask = torch.where(has_any, mask / denom.clamp_min(1), mask)
            att[:,:,self.v_shift+self.att_shift:] = mask
        
        elif self.method == "relumax":
            beta = torch.exp(
                torch.clamp(
                    self.kernel_beta * self.exp_param_scale,
                    max=self.kernel_beta_upbound
                )
            ).view(1, nh, 1, 1)   # <-- STOP HERE, NO EXPAND
            h = beta.reciprocal()        # bandwidth
            gamma = h.square()          # gamma = h^2
            b = 1.0                     # fixed margin
            # Use the raw (un-masked) attention and zero-out masked positions for numerical stability
            selected = raw.masked_fill(self.attn_mask[:, :, :T, :T], -1e12)
            selected = selected[:, :, self.v_shift + self.att_shift :]

            # compute non-negative scores and normalize; fall back to uniform on empty rows
            scores = selected - selected.max(dim=-1, keepdim=True).values
            scores = scores / gamma + b
            scores = F.relu(scores)
            row_sum = scores.sum(dim=-1, keepdim=True)

            # safe division: if sum==0, output 0
            weights = torch.where(row_sum > 0, scores / row_sum, torch.zeros_like(scores))
            att[:,:,self.v_shift+self.att_shift:] =weights
        
        elif self.method == "norm_relu":
            beta = torch.exp(
                torch.clamp(
                    self.kernel_beta * self.exp_param_scale,
                    max=self.kernel_beta_upbound
                )
            ).view(1, nh, 1, 1)   # <-- STOP HERE, NO EXPAND

            h = beta.reciprocal()         # safer than 1 / beta
            gamma = h.square().mul_(0.5) # in-place
            b = 1.0 - 2.0 * beta.square()  # since 1 - 2 / h^2 = 1 - 2 * beta^2   
            # 1) Zero out masked positions BEFORE ReLU
            selected = raw.masked_fill(self.attn_mask[:,:,:T,:T], 0.0)      
            selected = selected[:,:,self.v_shift+self.att_shift:]
            scores = selected / gamma + b
            scores = F.relu(scores)
            row_sum = scores.sum(dim=-1, keepdim=True)   # (B,H,Q,1)
            valid = valid[:,:,self.v_shift+self.att_shift:]
            valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            uniform = valid.to(scores.dtype) / valid_count
            weights = torch.where(
                    row_sum > 0,
                    scores / row_sum.clamp_min(1e-6),   # normal normalized ReLU
                    uniform             # indeterminate → uniform over valid
                )
            att[:,:,self.v_shift+self.att_shift:] = weights
        else:
            raise NotImplementedError(f"Unknown method {self.method}")
        att[:,:,:self.v_shift+self.att_shift] = 0 

        att = self.ic_dropout(att)
        y = att @ v 

       
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

         
        return y

class PersistentAssoMemBlock(nn.Module): # leakyaverage key, then normalize. convolution on value, then normalize. cheat first token
    
    def __init__(self,  n_head=12, n_embd=768, block_size=512, \
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


        self.att_shift = 0
        self.value_beta_init = -0.5 
        self.exp_param_scale = 10
       
        
        self.leaky_attn_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1,1,self.block_size,self.block_size) == 0

        self.pmem = Pmem(pmem_count, pmem_size, self.n_embd//self.n_head, self.n_head, ic_dropout)
        self.k_featurizer = FeatureExtractor(fe_type=k_fe_type, shift=0,            norm=False, \
                                            kernel_size=k_kernel_size, n_embd=self.n_embd, n_head = n_head, bias = self.bias)
        
        self.ic_dropout = nn.Dropout(ic_dropout)
        
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(hd_dropout)
            
        # # leakyavg coef 
        self.leakyavg_coef = torch.zeros(self.block_size, self.block_size)  #* float('-inf')
        for i in range(self.block_size):
            self.leakyavg_coef = torch.diagonal_scatter(self.leakyavg_coef, -torch.ones(self.block_size-i)*i, -i)

    
        self.leaky_key_beta = nn.Parameter( torch.linspace(0.5, 5, self.n_head) / self.exp_param_scale) 
        self.kernel_beta = nn.Parameter(torch.ones(self.n_head)/self.exp_param_scale) # for fp16, max float is 65536. kernel_beta < math.log(math.sqrt(65536)) ~5.5
        self.kernel_beta_upbound = 5 # assume fp16. 

        
        self.value_beta  = nn.Parameter(torch.ones(self.n_head) * (self.value_beta_init) / self.exp_param_scale)
        
       

    def forward(self, x):
        #print(x.dtype)
        B,T,C= x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        hs = self.n_embd // self.n_head
        nh = self.n_head


        k = self.k_featurizer(x)

        leaky_key_beta = self.leaky_key_beta.abs()*self.exp_param_scale
        leakyavg_coef = torch.exp(self.leakyavg_coef[:T,:T].view(1,1,T,T).expand(1, nh, T, T) \
                                    * leaky_key_beta.view(1,nh,1,1).expand(1,nh,T,T)) * \
                                    (1 - torch.exp(-leaky_key_beta)).view(1,nh,1,1).expand(1,nh, T, T)
        leakyavg_coef = leakyavg_coef.masked_fill(self.leaky_attn_mask[:,:,:T,:T], 0)
 
        k = leakyavg_coef.expand(B, nh, T, T) @ k 
        k = k / k.norm(dim=-1, keepdim=True)
        k = k *  (torch.exp(torch.clamp(self.kernel_beta*self.exp_param_scale, max=self.kernel_beta_upbound))**2 ).view(1,nh,1,1).expand(B, nh, T,hs)  #(B, nh, T, hs)
        # k times exp(kernel_beta)**2 because one scalar is for persistent memory 
        
        y = self.pmem(k) * (torch.exp(self.value_beta*self.exp_param_scale).view(1,nh,1,1).expand(B, nh, T,hs))
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))


        return y

class AttnOnlyBlock(nn.Module):
    def __init__(self, method="softmax", n_head=12, n_embd=768, v_shift=1, block_size=512, \
                k_fe_type='linearconv', v_fe_type='lowrlinearconv', \
                k_kernel_size=1, v_kernel_size=2, \
                ic_dropout=0.05, hd_dropout=0.05, 
                bias=False, 
                pmem_count=1, pmem_size=2688, pre_ln=True,attn_only=False, \
                config={}):
        super().__init__()
        self.attn_only = attn_only

        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.attn = InContextAssoMemBlock(n_head=n_head, method=method, n_embd=n_embd, v_shift=v_shift, block_size=block_size, \
                                            k_fe_type=k_fe_type, v_fe_type=v_fe_type, \
                                            k_kernel_size=k_kernel_size, v_kernel_size=v_kernel_size, \
                                            ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                                            bias=bias, \
                                            config=config)
        if not self.attn_only:
            self.ln2 = LayerNorm(n_embd, bias=bias)
            self.pmem = PersistentAssoMemBlock(n_head=n_head, n_embd=n_embd, block_size=block_size, \
                                                k_fe_type=k_fe_type, 
                                                k_kernel_size=k_kernel_size, 
                                                ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                                                bias=bias, \
                                                pmem_count=pmem_count, pmem_size = pmem_size,\
                                                config=config)
            
        self.pre_ln = pre_ln

    def forward(self, x):

        if self.pre_ln: # pre layernorm
            x = x + self.attn(self.ln1(x))
            if not self.attn_only:
                x = x + self.pmem(self.ln2(x))
        else:           # post layernorm
            x = self.ln1(x)
            x = x + self.attn(x)
            if not self.attn_only:
                x = self.ln2(x)
                x = x + self.pmem(x)
        return x

class StackAssoMem(nn.Module):
    def __init__(self, vocab_size, method="softmax", n_head=12, n_embd=768, n_layer=12, v_shift=1, block_size=512, \
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
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, self.n_embd)
        self.blocks = nn.Sequential(
            *[AttnOnlyBlock(n_head=n_head, method=method, n_embd=n_embd, v_shift=v_shift, block_size=block_size, \
                            k_fe_type=k_fe_type, v_fe_type=v_fe_type, \
                            k_kernel_size=k_kernel_size, v_kernel_size=v_kernel_size, \
                            ic_dropout=ic_dropout, hd_dropout=hd_dropout, 
                            bias=bias, 
                            pmem_count=pmem_count, pmem_size=pmem_size, pre_ln=pre_ln, attn_only=attn_only, \
                            config=config) for i in range(n_layer)]
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

        
        
    def forward(self, idx, targets=None, logits_only = False): # state=None,return_hidden_outputs=None):#, futures=None, storyid=None, skip_next=False, skip_future=False):  # fugures (b,n,t)
       
        device = idx.device
        b, t = idx.size()
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        #logits, loss =  None, 0
        logits_scale = 1.0 / math.sqrt(self.n_embd)

        if logits_only:
            logits = logits_scale * self.head(x[:,self.config.skip_tokens:])
            return logits

       
        if targets is not None: 
            logits = logits_scale * self.head(x[:,self.config.skip_tokens:])

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
                attn_mask = block.attn.attn_mask.to(device, **kwargs)
                leakyavg_coef =   block.attn.leakyavg_coef.to(device, **kwargs)
                leaky_attn_mask = block.attn.leaky_attn_mask.to(device, **kwargs)
           
            block.attn.leaky_attn_mask = leaky_attn_mask
            block.attn.attn_mask = attn_mask
            block.attn.leakyavg_coef = leakyavg_coef
            if not self.attn_only:
                block.pmem.leakyavg_coef = leakyavg_coef
                block.pmem.leaky_attn_mask = leaky_attn_mask
                
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

