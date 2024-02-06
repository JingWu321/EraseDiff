<div align="center">

# EraseDiff: Erasing Data Influence in Diffusion Models

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2301.11308&color=B31B1B)](https://arxiv.org/abs/2401.05779)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="./Images/teaser.png" width="80%">
  <br />
  <span>Figure 1: Generated samples by our method, EraseDiff, to erase the targeted class/concept. \algname can forget classes and avoid NSFW content.</span>
</p>


## Abstract
In this work, we introduce an unlearning algorithm for diffusion models. Our algorithm equips a diffusion model with a mechanism to mitigate the concerns related to data memorization. To achieve this, we formulate the unlearning problem as a constraint optimization problem, aiming to preserve the utility of the diffusion model on the remaining data and scrub the information associated with forgetting data by deviating the learnable generative process from the ground-truth denoising procedure. To solve the resulting problem, we adopt a first-order method, having superior practical performance while being vigilant about the diffusion process. Empirically, we demonstrate that our algorithm can preserve the model utility, effectiveness, and efficiency while removing across the widely-used diffusion models and in both conditional and unconditional image generation scenarios.

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
