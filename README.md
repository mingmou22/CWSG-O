# CWSG: Channel-Wise Structure-Guided Perturbations for Zero-Query Black-Box Adversarial Transfer

Official implementation of CWSG, a black-box adversarial attack that requires
**no queries to the victim model and no surrogate model**. Adversarial examples
are synthesized in the HSV color space, guided by a second image, with
perturbations propagated over a mask-gated patch graph through a fixed,
parameter-free Chebyshev spectral filter.

## Installation

```bash
pip install -r requirements.txt
```

The first run downloads the pre-trained weights of Mask R-CNN (foreground
masks) and VGG-19 (style objective) automatically; an internet connection is
required once.

## Quick Start

```bash
python attack.py --img path/to/source.png --target_img path/to/guidance.png
```

Outputs (adversarial image, comparisons, masks, per-channel perturbation maps,
loss traces, and the graph summary) are written to `cwsg_outputs/` by default.

To also score the adversarial example on built-in torchvision victims:

```bash
python attack.py --img source.png --target_img guidance.png --evaluate --victims resnet50 vgg16 vit_b_16
```

## Default Configuration

The defaults match the paper (Sec. 4.1): patch size 8, stride 4, filter order
K = 2, step size 0.1 with exponential decay, TV weight 0.2 with linear warm-up,
20 iterations, key-region fraction 0.3, channel weights [0.2, 0.3, 0.5], and a
per-channel HSV budget of 8/255 (circular metric on the hue channel).

```
python attack.py --help
```

lists all options.

## Code Layout

```
attack.py            entry point and optional victim evaluation
cwsg/config.py       configuration dataclass
cwsg/color.py        differentiable RGB<->HSV conversion and budget projection
cwsg/segmentation.py Mask R-CNN foreground masks (full-image fallback)
cwsg/graph.py        patch graph, betweenness centrality, fixed spectral filter
cwsg/losses.py       hue / style / frequency alignment losses and graph TV
cwsg/optimize.py     per-channel projected momentum optimization loop
cwsg/utils.py        seeding, I/O, and visualization helpers
```
