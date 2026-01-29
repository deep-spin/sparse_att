# sparse_attention_as_kernel_regression
# Sparse Attention as Compact Kernel Regression

Official code for the **Sparse Attention as Compact Kernel Rgression**. paper
*Saul Santos*, *Nuno Gonçalves*, *Daniel C. McNamee*, *Marcos Treviso*, and *André F.T Martins*

<p align="center">
  <img src="./overview.png" alt="Alt text" width="1000"/>
</p>
**Abstract**: *Recent work has revealed a link between self-attention mechanisms in transformers and test-time kernel regression via the Nadaraya–Watson estimator, with standard softmax attention corresponding to a Gaussian kernel.  
However, a kernel-theoretic understanding of *sparse* attention mechanisms is currently missing. In this paper, we establish a formal correspondence between sparse attention and *compact* (bounded-support) kernels. We show that normalized ReLU and sparsemax attention arise from Epanechnikov kernel regression under fixed and adaptive normalizations, respectively.  
More generally, we demonstrate that widely used kernels in nonparametric density estimation—including Epanechnikov, biweight, and triweight—correspond to $\alpha$-entmax attention with  
$\alpha = 1 + \frac{1}{n}$ for $n \in \mathbb{N}$, while the softmax/Gaussian relationship emerges in the limit $n \to \infty$. This unified perspective explains how sparsity naturally emerges from kernel design and provides principled alternatives to heuristic top-$k$ attention and other associative memory mechanisms. Experiments with a kernel-regression-based variant of transformers—**Memory Mosaics**—show that kernel-based sparse attention achieves competitive performance on language modeling, in-context learning, and length generalization tasks, offering a principled framework for designing attention mechanisms.
*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](to add) (arXiv)
