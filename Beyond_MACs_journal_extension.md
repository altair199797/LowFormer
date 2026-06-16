# Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![IJCV](https://img.shields.io/badge/IJCV-2026-blue.svg)](https://doi.org/10.1007/s11263-026-02873-5)

This is the official repository supplement for the journal article:

> **Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones**
> Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
> *International Journal of Computer Vision*, vol. 134, no. 6, pp. 295, 2026.
> [https://doi.org/10.1007/s11263-026-02873-5](https://doi.org/10.1007/s11263-026-02873-5)

This journal article extends the WACV 2025 conference paper ["LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones"](https://www.arxiv.org/pdf/2409.03460). All base LowFormer code, documentation, and pretrained models reside in the main repository — please refer to [README.md](README.md) for setup instructions, training, evaluation, and the original model zoo.

---

## Overview

The journal extension makes the following key contributions beyond the conference version:

- **Empirical analysis of MAC-based efficiency metrics**: We experimentally demonstrate that MAC counts are a poor predictor of actual execution time, in particular on edge devices. By contrasting MAC counts with measured latency and throughput across common architectural building blocks, we identify the principal factors that drive real-world hardware efficiency.
- **Design guidelines for hardware-efficient backbones**: Based on the above analysis, we provide actionable insights into macro- and micro-level design choices that lead to faster inference on edge GPU and desktop GPU hardware.
- **LowFormer Edge GPU model family (E1, E2, E3)**: Three new backbone variants are introduced whose micro design is explicitly informed by the hardware analysis. Compared to the base LowFormer family, the Edge GPU models replace or remove components that are expensive in practice despite having a low MAC footprint — most notably self-attention and MLP blocks — in favour of operations that map more efficiently to GPU execution units.

---

## Edge GPU Model Family: E1, E2, E3

Each E-series model is derived from a B-series counterpart by selectively removing components identified as inefficient on GPU hardware:

**Architecture:**

| Property | E1 | E2 | E3 |
| :--- | :---: | :---: | :---: |
| Based on | LowFormer-B1.5 | LowFormer-B3 | LowFormer-B3 |
| Channel widths | [20, 40, 80, 160, 320] | [32, 64, 128, 256, 512] | [32, 64, 128, 256, 512] |
| Stage depths | [0, 1, 1, **4**, **4**] | [1, 2, 3, **4**, **4**] | [1, 2, 3, 6, 6] |
| Self-attention removed | ✓ | ✓ | ✗ |
| MLP blocks removed | ✓ | ✓ | ✓ |

**ImageNet-1K results** (GPU Throughput: Nvidia A40, batch 200; GPU/TX2 Latency in ms, batch 1):

| Model | MACs (M) | GPU Throughput ↑ | GPU Latency ↓ | TX2 Latency ↓ | Top-1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LowFormer-E1 | 1350 | 6337 | 1.0 | 6.2 | 78.8 |
| LowFormer-E2 | 3800 | 2070 | 1.5 | 14.7 | 81.6 |
| LowFormer-E3 | 5350 | 1566 | 3.6 | 25.0 | 83.0 |

**Design changes relative to the B-series:**

- **E1 (based on B1.5)**: Self-attention and MLP blocks are removed throughout, and the depth of the two later stages is reduced from 6 to 4. The result is a purely convolutional backbone at the B1.5 capacity level, with a lower practical latency than the MAC count alone would suggest.
- **E2 (based on B3)**: The same attention-free and MLP-free design as E1, applied at the larger B3 capacity. Stage depths in the final two stages are likewise reduced from 6 to 4.
- **E3 (based on B3)**: Identical macro structure to B3 (same widths and depths), with only the MLP blocks removed. Self-attention is retained, making E3 the highest-accuracy variant of the E-series.

Pretrained checkpoints and detailed quantitative results are reported in the paper linked above. Checkpoints will be made available via the same [Dropbox link](https://www.dropbox.com/scl/fo/xtgv7fpae4vzpdu2ajsz1/ALuycdfNrmZ44yYCeE6ILPA?rlkey=2gfcrsryep8hnipw831ufymms&dl=0) as the base models once released.

---

## Usage

The Edge GPU models are accessible through the same `lowformer_model.py` standalone file used for the base LowFormer variants. Configuration files are provided under `configs/cls/imagenet/`.

```python
import torch
from lowformer_model import get_lowformer_E1, get_lowformer_E2, get_lowformer_E3

model = get_lowformer_E1(pretrained=True)
# model = get_lowformer_E2(pretrained=True)
# model = get_lowformer_E3(pretrained=True)

inp = torch.randn(1, 3, 224, 224)
out = model(inp)  # -> [1, 1000]
```

Refer to [README.md](README.md) for environment setup, ImageNet evaluation, throughput/latency benchmarking, and downstream task adaptation.

---

## Citation

If this journal extension is useful in your research, please cite:

```bibtex
@article{Nottebaum2026BeyondMACs,
  author  = {Nottebaum, Moritz and Dunnhofer, Matteo and Micheloni, Christian},
  title   = {Beyond {MAC}s: Hardware Efficient Architecture Design for Vision Backbones},
  journal = {International Journal of Computer Vision},
  year    = {2026},
  volume  = {134},
  number  = {6},
  pages   = {295},
  doi     = {10.1007/s11263-026-02873-5},
  url     = {https://doi.org/10.1007/s11263-026-02873-5}
}
```

If you use the base LowFormer models or codebase, please also cite the original conference paper (see [README.md § Citation](README.md#citation)).
