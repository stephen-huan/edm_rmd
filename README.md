## Randomized midpoint method

Implementation of the [randomized midpoint
method](http://arxiv.org/abs/1909.05503) for diffusion model sampling.

This code implements the score and relative score choices for
the linear factor (the so-called "SDE-adapted" choices, see
C.2.2 "Concrete choices of scaling factor" of the paper). For
the "network-adapted" denoiser and skip connection choices, see
the [nn-adapted](https://github.com/stephen-huan/edm_rmd/tree/nn-adapted) tag.

## License

This repository is forked from [edm](https://github.com/NVlabs/edm) by Tero
Karras, Miika Aittala, Timo Aila, and Samuli Laine. The contents of that
repository (including source code and pre-trained models) are licensed
under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
In this repository, `generate.py` has been modified to implement the
randomized midpoint method and `example.py` has been modified to work
with the latest Pillow.

### Upstream License

Copyright &copy; 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

`baseline-cifar10-32x32-uncond-vp.pkl` and `baseline-cifar10-32x32-uncond-ve.pkl` are derived from the [pre-trained models](https://github.com/yang-song/score_sde_pytorch) by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. The models were originally shared under the [Apache 2.0 license](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE).

`baseline-imagenet-64x64-cond-adm.pkl` is derived from the [pre-trained model](https://github.com/openai/guided-diffusion) by Prafulla Dhariwal and Alex Nichol. The model was originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).

`imagenet-64x64-baseline.npz` is derived from the [precomputed reference statistics](https://github.com/openai/guided-diffusion/tree/main/evaluations) by Prafulla Dhariwal and Alex Nichol. The statistics were
originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).
