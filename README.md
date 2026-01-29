# Sparse Attention as Compact Kernel Regression

Official code for the **Sparse Attention as Compact Kernel Regression** paper.

*Saul Santos*, *Nuno Gonçalves*, *Daniel C. McNamee*, *Marcos Treviso*, and *André F.T Martins*

<p align="center">
  <img src="./overview.png" alt="Alt text" width="1000"/>
</p>
**Abstract**: * Recent work has revealed a link between self-attention mechanisms in transformers and test-time kernel regression via the Nadaraya–Watson estimator, with standard softmax attention corresponding to a Gaussian kernel.  
However, a kernel-theoretic understanding of *sparse* attention mechanisms is currently missing. In this paper, we establish a formal correspondence between sparse attention and *compact* (bounded-support) kernels. We show that normalized ReLU and sparsemax attention arise from Epanechnikov kernel regression under fixed and adaptive normalizations, respectively.  
More generally, we demonstrate that widely used kernels in nonparametric density estimation—including Epanechnikov, biweight, and triweight—correspond to $\alpha$-entmax attention with  
$\alpha = 1 + \frac{1}{n}$ for $n \in \mathbb{N}$, while the softmax/Gaussian relationship emerges in the limit $n \to \infty$. This unified perspective explains how sparsity naturally emerges from kernel design and provides principled alternatives to heuristic top-$k$ attention and other associative memory mechanisms. Experiments with a kernel-regression-based variant of transformers—**Memory Mosaics**—show that kernel-based sparse attention achieves competitive performance on language modeling, in-context learning, and length generalization tasks, offering a principled framework for designing attention mechanisms.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](to add) (arXiv)

  All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

## Language Modeling
### Installation and Reproducibility
Follow [Memory Mosaics](https://github.com/facebookresearch/MemoryMosaics) instructions with our code and use the hyparparameters mentioned in the Appendix.

You will need to:

```
cd Library
pip install -r requirements.txt
```

## In-Context Learning
### Installation
Follow [Memory Mosaics](https://github.com/facebookresearch/MemoryMosaics) instructions. 
You will need to:

```
cd ICLL
pip install -r requirements.txt
```
If you have problems with triton in adasplash package (which you have also to install), remove this line of code (line 3) of the init file of adasplash library:

```
from .triton_entmax import triton_entmax 
```
### Reproducibility
Simply run this command for all method with these tuned hyperparameters:
```
for n in 1000 2500 5000 10000 20000 40000; do python train.py train.test=True experiment=dfa/mm dataset.num_test_examples=1000 dataset.num_examples=$n model.n_layer=2 model.method=softmax model.n_embd=128 model.n_head=8 optimizer.lr=1e-4 optimizer.weight_decay=0.01; done

```
