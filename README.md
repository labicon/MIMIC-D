# MIMIC-D: Multi-modal Imitation for MultI-agent Coordination with Decentralized Diffusion Policies

In this project we introduce **MIMIC-D**, a CTDE (Centralized Training, Decentralized Execution) framework that learns **decentralized diffusion policies** from multi-agent expert demonstrations to recover diverse, coordinated behaviors without explicit inter-agent communication.

[Paper](<https://arxiv.org/abs/2509.14159>)          
[Website](<https://iconlab.negarmehr.com/MIMIC-D/>)


## Overview

Many real-world multi-agent tasks have **multiple valid coordination modes** (e.g., pass-left vs pass-right) and cannot assume reliable centralized planners or explicit communication. MIMIC-D trains policies jointly with full information, then **executes with only local observations**, enabling **implicit coordination** while preserving **multi-modality** in the learned behaviors. We validate MIMIC-D in multiple simulation environments and on a **bimanual hardware** setup with heterogeneous arms (Kinova3 + xArm7).
<p align="center">
  <video width="600" controls muted playsinline preload="metadata">
    <source src="docs/videos/MIMIC-D.mp4" type="video/mp4" />
    Your browser doesn’t support HTML5 video.
  </video>
</p>




## Project Organization

- `dependencies/` — conda environment file (it may be easier to simply install dependencies as you go)
- `lift/` — simulated two-arm pot lifting experiment in robosuite
- `lift_hardware/` — two-arm pot lifting experiment on Kinova3 and xArm7 on hardware
- `three_agent_road/` — three agent road crossing environment
- `two_agent_swap/` — two agent swap environment
- `docs/` — all the elements to build the project website


## Getting Started (fill in after release)

1. _TODO: environment setup (conda, CUDA/cuDNN, PyTorch version, robosuite, etc.)_
2. _TODO: data preparation (where to download / how to format expert demos)_
3. _TODO: training (commands & key flags)_
4. _TODO: sampling / evaluation (receding-horizon execution, metrics, plotting)_


## Citation

If you use MIMIC-D, please cite:

```bibtex
@article{dong2025mimic,
  title={MIMIC-D: Multi-modal Imitation for MultI-agent Coordination with Decentralized Diffusion Policies},
  author={Dong, Dayi and Bhatt, Maulik and Choi, Seoyeon and Mehr, Negar},
  journal={arXiv preprint arXiv:2509.14159},
  year={2025}
}
```


## Acknowledgments

Our diffusion transformer architecture is largely based on the [AlignDiff code](https://github.com/ZibinDong/AlignDiff-ICLR2024/tree/main).