<div align="center">

# Erasing Undesirable Influence in Diffusion Models

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2301.11308&color=B31B1B)](https://arxiv.org/abs/2401.05779)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="./Images/teaser.png" width="80%">
  <br />
  <span>Figure 1: Generated samples by our method, EraseDiff, to erase the targeted class/concept. EraseDiff can forget classes and avoid NSFW content.</span>
</p>


## Abstract
Diffusion models are highly effective at generating high-quality images but pose risks, such as the unintentional generation of NSFW (not safe for work) content. Although various techniques have been proposed to mitigate unwanted influences in diffusion models while preserving overall performance, achieving a balance between these goals remains challenging. In this work, we introduce EraseDiff, an algorithm designed to preserve the utility of the diffusion model on retained data while removing the unwanted information associated with the data to be forgotten. Our approach formulates this task as a constrained optimization problem using the value function, resulting in a natural first-order algorithm for solving the optimization problem. By altering the generative process to deviate away from the ground-truth denoising trajectory, we update parameters for preservation while controlling constraint reduction to ensure effective erasure, striking an optimal trade-off. Extensive experiments and thorough comparisons with state-of-the-art algorithms demonstrate that EraseDiff effectively preserves the model’s utility, efficacy, and efficiency.

## Getting Started
The code is split into two subfolders, i.e., DDPM and Stable Diffusion experiments. Detailed instructions are included in the respective subfolders.

## BibTeX
```
@article{wu2024erasediff,
  title={EraseDiff: Erasing Data Influence in Diffusion Models},
  author={Wu, Jing and Le, Trung and Hayat, Munawar and Harandi, Mehrtash},
  journal={arXiv preprint arXiv:2401.05779},
  year={2024}
}
```
## Acknowledgements
This repository makes liberal use of code from [ESD](https://github.com/rohitgandikota/erasing/tree/main), [Selective Amnesia](https://github.com/clear-nus/selective-amnesia) and [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency).
