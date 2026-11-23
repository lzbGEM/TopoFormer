# TopoFormer

## A Multi-Scale Topology-Aware Vision Transformer for H&E Tumor Classification

TopoFormer is a topology-aware histopathology classification framework that integrates a pretrained **DINOv3 ViT-7B/16** visual foundation model with **persistent-homology (PH)** representations. The framework combines a **multi-scale topology feature pyramid (TopoFPN)**, **topology-conditioned token interaction (TopoCTI)**, **ConvNeXtV2-based gated fusion**, **discriminative–nuisance feature disentanglement (DNFD)**, and a **global topological vector** for patch-level H&E tumor classification.

<p align="center">
  <img src="docs/figures/Architecture.png" alt="TopoFormer architecture" width="95%">
</p>

**Fig. 1.** Overall architecture of TopoFormer. TopoFPN fuses PH maps at four window sizes. TopoCTI passes the four fused maps to the successive stages of DINOv3, and global average pooling of the same native maps yields a 1,600-dimensional topology vector. ConvNeXtV2 and DINOv3 features are combined by gated fusion. DNFD supervises the fused visual representation during training. Logistic regression classifies the normalized visual representation and topology vector.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment and Installation](#environment-and-installation)
- [Pretrained Weights](#pretrained-weights)
- [Datasets](#datasets)
- [Data Partitions and Experimental Protocol](#data-partitions-and-experimental-protocol)
- [TCGA-UT External Evaluation](#tcga-ut-external-evaluation)
- [Data Preparation](#data-preparation)
- [Persistent-Homology Preprocessing](#persistent-homology-preprocessing)
- [Architecture and Training Configuration](#architecture-and-training-configuration)
- [DINOv3 Backbone Modification](#dinov3-backbone-modification)
- [Core Modules](#core-modules)
- [Training](#training)
- [Final Feature-Fusion Evaluation](#final-feature-fusion-evaluation)
- [Reported Benchmark Results](#reported-benchmark-results)
- [Additional Analyses](#additional-analyses)
- [Computational Considerations](#computational-considerations)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [Third-Party Resources and Dataset Citations](#third-party-resources-and-dataset-citations)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Research Use and Reporting](#research-use-and-reporting)
- [Contact](#contact)

---

## Overview

TopoFormer is designed for classification of H&E-stained histopathology patches. It combines high-capacity pretrained visual representations with explicit topological descriptors derived from persistent homology.

The released implementation contains the following principal components:

- **DINOv3 ViT-7B/16 visual backbone** for pretrained visual representation learning.
- **LoRA adaptation** for parameter-efficient adaptation of the frozen DINOv3 backbone.
- **Persistent-homology encoder** using grayscale foreground point sets, alpha complexes, and H1 persistence.
- **TopoFPN** for organizing local PH descriptors into aligned multi-scale topology maps.
- **TopoCTI** for topology-conditioned interaction with DINOv3 patch tokens.
- **ConvNeXtV2-based gated fusion** for complementary convolutional visual representation.
- **DNFD** for encouraging task-related feature differentiation through discriminative and auxiliary representations with regularization objectives.
- **Global topological vector** for complementary sample-level topological information.
- **L2-regularized logistic regression** for the final feature-fusion classifier.
- **Reproducible preprocessing, training, feature export, and classification** for DeepHisto and CRC.
- **External TCGA-UT protocol and cohort manifest** for three-class glioma generalization.

The reported experiments concern **patch-level classification**. The current repository does not claim a validated whole-slide clinical deployment pipeline.

---

## Repository Structure

```text
TopoFormer/
├── prepare_ph.py           # Four-scale PH extraction and TopoFPN caching
├── run_topoformer.py       # Model validation, training and feature export
├── report.py               # Train-fit normalization, logistic regression and metrics
├── materials.py            # Configuration and PH cache conversion
├── ph/                     # Alpha complex, persistence images and TopoFPN
├── topoformer/
│   ├── data/               # Dataset splits, image loading and topology caches
│   ├── engine/             # Training setup, CTI preflight and feature export
│   └── model/              # TopoCTI, LoRA, ConvNeXtV2 fusion and DNFD
├── Backbone/dinov3/        # Patched DINOv3 backbone sources and upstream license
├── assets/                 # Fixed benchmark splits and TCGA-UT cohort manifest
├── docs/figures/           # Manuscript Figs. 1 and 2
├── config.example.json
├── requirements.txt
└── LICENSE
```

### Main entry points

From the repository root, create the local configuration and set the dataset and checkpoint paths.

```bash
cp config.example.json config.local.json
python prepare_ph.py --help
python run_topoformer.py --help
python report.py --help
```

Use the [preprocessing](#dataset-specific-topology-preprocessing), [training](#training), and [classification](#final-feature-fusion-evaluation) commands in order. The implementation and example commands cover the full TopoFormer model. The paper and supplement document the separate analysis protocols.

---

## Environment and Installation

### Reported environment

The reference software and hardware for the reported experiments and the public release are listed below.

| Package / Hardware | Version |
|---|---|
| Python | 3.11 |
| PyTorch / torchvision | 2.4.0 / 0.19.0 |
| CUDA build of PyTorch | 12.1 |
| GUDHI | 3.10.0 in the manuscript protocol; 3.11.0 pinned for the release |
| GPU | NVIDIA A40, 48 GB |

### Create the environment

```bash
conda create -n topoformer python=3.11 -y
conda activate topoformer
```

Install project dependencies:

```bash
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Verify the main software stack:

```bash
python -c "import torch, gudhi; print(torch.__version__, torch.version.cuda, gudhi.__version__)"
```

`requirements.txt` is a project-level dependency list. It records exact versions for selected packages, including PyTorch, torchvision, and GUDHI, and version bounds for other dependencies, including NumPy, Pillow, OpenCV, and tqdm. The original PH caches used in the manuscript were extracted with GUDHI 3.10.0. The public release specifies GUDHI 3.11.0.

If the default PyTorch wheel does not match the local CUDA installation, install the appropriate PyTorch/torchvision build for the target system before installing the remaining dependencies.

---

## Pretrained Weights

TopoFormer depends on pretrained DINOv3 and ConvNeXtV2 checkpoints. These files are not redistributed by this repository and remain subject to their original licenses and distribution terms.

### DINOv3

Official repository:

- https://github.com/facebookresearch/dinov3

Checkpoint used in the reported configuration:

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

Set `dinov3` in `config.local.json` to the downloaded checkpoint.

### ConvNeXt V2

Official repository:

- https://github.com/facebookresearch/ConvNeXt-V2

Checkpoint used in the reported configuration:

```text
convnextv2_base_1k_224_ema.pt
```

Set `convnext` in the same configuration to the ConvNeXtV2-Base checkpoint.

---

## Datasets

| Dataset | Source | Role in this project |
|---|---|---|
| **DeepHisto** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.7941080) | Five-class brain histopathology classification |
| **NCT-CRC-HE-100K** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.1214456) | Nine-class colorectal tissue training/development set |
| **CRC-VAL-HE-7K** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.1214456) | Independent CRC final test set containing 7,180 patches |
| **TCGA-UT** | [Zenodo](https://doi.org/10.5281/zenodo.5889558) · [Hugging Face](https://huggingface.co/datasets/dakomura/tcga-ut) | External three-class glioma evaluation |

Users are responsible for downloading each dataset from its original source and complying with the corresponding license, access, attribution, and usage requirements.

---

## Data Partitions and Experimental Protocol

### DeepHisto

- The official **patient-wise training/test partition** is preserved.
- The held-out official test set contains **4,174 patches**.
- A stratified **10% of the official training set** is reserved for internal validation.
- The remaining **90% of the official training set** is used for model training.
- The official test set is not used for checkpoint selection.

Reported training seeds:

```text
42, 123, 666
```

### CRC

- **NCT-CRC-HE-100K** supplies the training/development samples.
- A stratified **10% of NCT-CRC-HE-100K** is reserved for internal validation.
- The remaining **90%** is used for training.
- **CRC-VAL-HE-7K** is retained as the independent final test set.
- CRC-VAL-HE-7K is not used for checkpoint selection.

Reported training seeds:

```text
12, 42, 666
```

### Checkpoint selection

For both datasets, checkpoints are selected according to:

```text
best validation balanced accuracy
```

Foreground thresholds were selected by mean validation balanced accuracy across the three seeds. The official test partitions are reserved for final evaluation. DeepHisto's internal holdout is patch-level because patient identifiers are unavailable within the released training data.

---

## TCGA-UT External Evaluation

The external evaluation uses **786 TCGA-UT patches** from 83 TCGA case identifiers and 187 slide identifiers. Labels use the simplified 2021 classification in the subtype annotation file.

Importantly, this experiment uses **separately trained three-class DeepHisto models**. Training includes astrocytoma, glioblastoma, and oligodendroglioma only; the normal and necrosis classes are excluded. The selected checkpoints and the late-fusion classifier were fitted using the three-class DeepHisto development data.

| External directory | Three-class label | Patches | Distinct WSI identifiers |
|---|---|---:|---:|
| `astrocytoma/` | `astro` | 282 | 69 |
| `glioblastoma/` | `gbm` | 344 | 79 |
| `oligodendroglioma/` | `oligo` | 160 | 39 |
| **Total** |  | **786** | **187** |

The cohort is tracked with:

```text
assets/TCGA_UT_786_sample_manifest.csv
```

The manifest contains:

```text
Dataset
OriginalFolder
MappedLabel5to3
WSI_ID
PatchPath
```

`WSI_ID` identifies a source slide directory. Case and slide counts are reported separately above.

Original TCGA-UT imagery can be obtained from:

- https://doi.org/10.5281/zenodo.5889558
- https://huggingface.co/datasets/dakomura/tcga-ut

The released [786-patch manifest](assets/TCGA_UT_786_sample_manifest.csv) gives paths relative to each external image root. Preserve this folder and filename layout under both conditions.

### External stain normalization

External images are evaluated under two conditions:

1. **Original external images**
2. **Reinhard-style color normalization**

Reference BGR-channel means and standard deviations are estimated from **3,000 sampled DeepHisto training patches**. These statistics are applied only during external evaluation.

The trained model checkpoints remain fixed across the two external-evaluation conditions. The external protocol and results are described in the manuscript. The cohort manifest is supplied for reference.

---

## Data Preparation

### DeepHisto

Download DeepHisto from the official Zenodo record and preserve its official patient-wise train/test organization.

A simplified local layout is:

```text
/path/to/DeepHisto/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Set `datasets.briancancer.train` and `datasets.briancancer.test` to the directories that contain the class folders. The manifest prefixes for these locations are `/datasets/Briancancer/train/train` and `/datasets/Briancancer/test/test`.

### CRC

A simplified local layout is:

```text
/path/to/CRC/
├── NCT-CRC-HE-100K/
└── CRC-VAL-HE-7K/
```

Set `datasets.crc.train` and `datasets.crc.test` to the two directories above. The supplied split files retain the exact patch order for both datasets.

### Large generated files

Datasets and the following generated files should normally be stored outside the Git repository:

- pretrained model checkpoints;
- TopoFPN `.npz` archives;
- cached feature matrices;
- model checkpoints;
- experiment logs;
- external-evaluation image copies;
- large intermediate PH outputs.

Set explicit local paths in `config.local.json`. Update `path_roots` so the manifest paths under `/datasets` resolve to the corresponding local image directories.

---

## Persistent-Homology Preprocessing

The visual and topological branches use separate input preprocessing.

### Visual branch

The visual branch uses:

```text
224 × 224 RGB
```

with ImageNet normalization and no random data augmentation.

### PH branch

The PH branch uses separately resized grayscale images.

| Configuration | DeepHisto | CRC |
|---|---:|---:|
| PH grayscale input | **128 × 128** | **224 × 224** |
| Foreground threshold | **200** | **180** |
| Grayscale range | 8-bit | 8-bit |
| Morphological dilation | 2 × 2 kernel, 1 iteration | Same |
| Local PH windows | 16, 32, 64, 128 px | Same |
| Stride | Equal to window size | Same |
| Native grids (L0 to L3) | 8 × 8, 4 × 4, 2 × 2, 1 × 1 | 14 × 14, 7 × 7, 3 × 3, 1 × 1 |
| Complex | Alpha complex | Same |
| Homology dimension | **H1** | Same |
| Persistence image | 20 × 20 | Same |
| Gaussian bandwidth | 1.0 | Same |
| Persistence weighting | Uniform | Same |
| TopoFPN alignment | Nearest-neighbor top-down upsampling | Same |

### Foreground construction

The PH preprocessing procedure is:

1. Convert the source RGB image to grayscale with OpenCV.
2. Resize the grayscale image using area interpolation.
3. Apply the dataset-specific foreground threshold.
4. Pixels with intensities **above** the threshold are set to zero.
5. Intensities of retained pixels are preserved.
6. Apply one morphological dilation using a `2 × 2` kernel.
7. Within each local window, collect the spatial coordinates of nonzero pixels.
8. Treat these coordinates as the point set used to construct an alpha complex.
9. Compute **H1 persistent homology**.
10. Convert the persistence representation into a `20 × 20` persistence image.
11. Organize local descriptors into four multi-scale spatial maps.
12. Fuse the maps through the TopoFPN top-down pathway.

Only complete windows are extracted, starting from the upper-left corner without padding. On CRC, 64- and 128-pixel windows cover the upper-left 192 × 192 and 128 × 128 regions, respectively. The encoder removes the (0.25, 0.5) birth–death pair and fits a birth–persistence grid separately for each nonempty window diagram. An empty diagram produces an all-zero image.

### Dataset-specific topology preprocessing

```bash
python prepare_ph.py briancancer 200 --config config.local.json
python prepare_ph.py crc 180 --config config.local.json
```

The extractor saves the four TopoFPN levels on their native grids for token interaction and the **1,600-dimensional** global vector formed from the same native fused levels. The three foreground thresholds documented in Supplementary Table S3 are supported as preprocessing settings. Each threshold has its own cache location.

A typical cache contains:

```text
prepared/briancancer/200/
├── maps_L0.mmap/
├── maps_L1.mmap/
├── maps_L2.mmap/
├── maps_L3.mmap/
└── gv4.npz
```

Set each dataset’s `maps` to the prefix before `_L0` and `pooled` to the four-scale vector NPZ. Existing compatible map NPZ archives can be read directly or unpacked with `python materials.py /path/to/cache/prefix --compact`. Generate a four-scale GV before scoring older runs with legacy pooled data.

### Cache alignment requirement

The image files, split definitions, and topology-cache entries must refer to the same samples.

Before training, verify that:

```text
image identifier
↔ split entry
↔ cached PH / TopoFPN entry
```

are aligned consistently.

Incorrect cache-to-image matching invalidates the corresponding experiment even if the array dimensions remain compatible.

---

## Architecture and Training Configuration

| Component | Reported configuration |
|---|---|
| Visual backbone | DINOv3 ViT-7B/16 |
| Backbone initialization | Official pretrained weights |
| DINOv3 backbone | Frozen |
| Adaptation | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| TopoCTI stages | Blocks 1–10, 11–20, 21–30, 31–40 |
| TopoCTI injection locations | Before blocks 1, 11, 21, and 31 |
| TopoCTI interaction | Spatially aligned patch and topology projections, 512 channels, FP32 |
| Patch grid | Native topology grid bilinearly resampled to 14 × 14, `align_corners=False` |
| Residual update | Norm limit of 0.3 × reference-token norm, followed by a learned scale of 0.1 × sigmoid(gate) |
| CLS update | Learned spatial pooling at stage 4 |
| Global topology vector | GAP on four fused native maps, 4 × 400 = 1,600 dimensions |
| ConvNeXtV2 branch | Pretrained and frozen |
| ConvNeXtV2 role | Gated complementary visual fusion |
| DNFD `λortho` | 0.05 |
| DNFD `λadv` | 0.1 |
| Optimizer | AdamW |
| Learning rate | 2.5 × 10⁻⁵ in epoch 1, 5 × 10⁻⁵ in epoch 2 |
| Weight decay | 0.05 |
| Epochs | 2 |
| Initial batch size | 8 |
| Mixed precision | FP16 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| EMA | 0.999 |
| Sampling | Class-balanced sampling with replacement |
| Checkpoint criterion | Validation balanced accuracy |
| Final feature classifier | L2-regularized logistic regression |
| Logistic-regression `C` | 1.0 |

### Final feature representation

The `feat` field contains the visual feature after DINOv3–ConvNeXtV2 gated fusion. The scorer standardizes this feature with training-set means and standard deviations. It min–max scales the global PH vector using the same training split. Each branch is then L2-normalized. The classifier receives their concatenation with a weight of 0.3 on the topology branch. The final logistic regression uses balanced class weights and `C = 1.0`.

### DNFD interpretation

DNFD is designed to encourage **task-related feature differentiation** through a discriminative branch, an auxiliary branch, and regularization objectives.

The current experimental evidence supports partial class-related representation differentiation. It should **not** be interpreted as establishing complete removal of stain, background, or other nuisance information.

---

## DINOv3 Backbone Modification

This repository includes a local DINOv3 source tree under:

```text
Backbone/dinov3/
```

The project is based on the official DINOv3 implementation:

- https://github.com/facebookresearch/dinov3

The principal local DINOv3 source file modified for TopoFormer is:

```text
Backbone/dinov3/dinov3/dinov3/models/vision_transformer.py
```

The transformer calls the hook before the first block of each stage. The hook returns the updated token stream in the same list position so that the next block receives it. The training preflight checks all four handoffs and verifies that swapping inputs at each topology level changes the output. The training loop checks gradients and parameter updates in the patch projections and final CLS projection.

Users comparing this repository against upstream DINOv3 should therefore distinguish:

1. unchanged upstream DINOv3 source files;
2. the TopoFormer-specific modification to `vision_transformer.py`;
3. TopoFormer modules implemented under `topoformer/`.

---

## Core Modules

### 1. TopoFPN

TopoFPN builds one map of 400-channel persistence images per window size. Starting with T3 = F̂3, it adds the nearest-neighbor upsampled coarser map to each finer map, Tl = F̂l + Up(Tl+1). Fusion matches channels by their within-window persistence-image grid indices. A channel can correspond to different absolute birth and persistence values in different windows because the grid bounds are fitted separately.

The same fused maps support TopoCTI and the global topology vector. See [the PH and TopoFPN implementation](ph/features.py).

### 2. TopoCTI

TopoCTI injects progressively broader topology into deeper visual stages. The four stage-specific maps L0–L3 enter immediately before blocks 1, 11, 21 and 31, respectively. Each stage spans ten DINOv3 blocks.

<p align="center">
  <img src="docs/figures/TopoCTI.png" alt="Four stages of TopoCTI, local patch gates and final CLS pooling" width="95%">
</p>

At each stage, the module bilinearly resamples the native topology grid to the 14 × 14 patch grid. Separate layer normalizations and linear maps project the visual token and its aligned topology vector into 512 dimensions. A sigmoid of the visual projection gates the topology channels element by element. A linear map projects this interaction back to the token dimension. The update is clipped to 30% of the reference-token norm and multiplied by a learned residual scale initialized to 0.05.

**Fig. 2.** Four-stage injection, position-wise gated patch updates, and the stage-4 CLS update.

Only the last stage also updates CLS. A learned scalar score per patch gives softmax weights over the 196 interactions. Their weighted sum is projected, clipped against the CLS norm and added with its own learned scale. The four register tokens pass through each stage unchanged. See [the stage implementation](topoformer/model/four_stage_cti.py) and [model assembly](topoformer/model/model_builder.py).

### 3. LoRA adaptation

LoRA adapts the combined query–key–value projection and the output projection of self-attention in all 40 ViT-7B/16 blocks. The original attention weights and the feed-forward layers remain frozen.

Implementation:

```text
topoformer/model/lora.py
```

Reported settings:

```text
rank    = 16
alpha   = 16
dropout = 0.05
```

### 4. DNFD

DNFD penalizes cross-covariance between the discriminative and auxiliary features. It supervises the DFDA classifier with class labels and encourages the NFDA decoder to output a uniform class distribution.

Implementation:

```text
topoformer/model/disentanglement.py
```

DFDA produces the discriminative features and NFDA the nuisance features. Their supervision shapes the shared visual representation during training. The export retains both branches and their predictions. The cross-covariance regularization is active from epoch 1. The auxiliary KL term has zero weight during epoch 1 and becomes active in epoch 2.

Reported objective weights:

```text
lambda_ortho = 0.05
lambda_adv   = 0.1
```

### 5. ConvNeXtV2-based gated fusion

A frozen pretrained ConvNeXtV2-Base branch provides complementary convolutional visual features.

Implementation:

```text
topoformer/model/convnext_fusion.py
```

The ConvNeXtV2 representation is integrated through a learned gated fusion mechanism rather than simple unweighted concatenation.

### 6. Global topological vector

Each native fused map from the 16-, 32-, 64- and 128-pixel windows is spatially averaged once. Concatenating GAP(T0), GAP(T1), GAP(T2), and GAP(T3) in level order yields four 400-dimensional blocks. This uses the actual native cell counts, including the 3 × 3 and 1 × 1 grids for the coarser CRC scales.

For the manuscript's **Global Vector Only** branch-control analysis, L2-regularized logistic regression used the topology vector as its sole input.

---

## Training

Prepare the PH cache and set the dataset paths, split manifest, pretrained weights and PH paths in `config.local.json`. Choose a fresh output directory for each seed. The examples use one visible CUDA GPU. The optional `preflight` action accepts the same arguments as `train` and checks that each topology level reaches the DINOv3 token stream.

### DeepHisto example

```bash
CUDA_VISIBLE_DEVICES=0 python run_topoformer.py train briancancer 200 42 runs/deephisto/seed42
```

DeepHisto uses threshold 200 and seeds 42, 123 and 666.

### CRC example

```bash
CUDA_VISIBLE_DEVICES=0 python run_topoformer.py train crc 180 42 runs/crc/seed42
```

CRC uses threshold 180 and seeds 12, 42 and 666. Set `--workers 0` to use the main process for image loading, for example when debugging a local data installation.

### Validation and test-set separation

Each epoch is evaluated with EMA weights on the stratified validation split. Feature export uses the ordinary weights saved at the epoch with the highest validation balanced accuracy. This leaves the designated official test set independent of epoch and threshold selection. A compact checkpoint omits the frozen DINOv3 parameters and verifies the original backbone checksum before reloading.

---

## Final Feature-Fusion Evaluation

```bash
RUN=runs/deephisto/seed42
CUDA_VISIBLE_DEVICES=0 python run_topoformer.py export briancancer 200 42 "$RUN" --split train val test
python report.py briancancer 200 42 "$RUN" --split val test
```

For CRC, use the corresponding run directory and replace `briancancer 200 42` with the dataset, threshold and seed used for training. Export only missing splits to complete a partial run. The validation and test predictions and metrics are saved under `scores/{val,test}/`. Match the seed, checkpoint, feature cache, topology cache, split manifest and test image paths for each run. The evaluation command fits the normalization and logistic regression on the training split and checks the identities and class labels of the selected PH rows.

---

## Reported Benchmark Results

The following results are reported as **mean ± sample standard deviation (%) over three seed-specific runs**.

| Dataset | Accuracy | Balanced accuracy | Macro precision | Macro specificity (OvR) | Macro F1 | Macro AUROC (OvR) |
|---|---:|---:|---:|---:|---:|---:|
| DeepHisto | 97.23 ± 0.36 | 86.78 ± 2.41 | 96.11 ± 0.24 | 98.87 ± 0.19 | 89.70 ± 2.18 | 99.85 ± 0.05 |
| CRC | 97.06 ± 0.24 | 95.69 ± 0.78 | 95.98 ± 0.33 | 99.64 ± 0.03 | 95.67 ± 0.61 | 99.80 ± 0.04 |

For CRC, balanced accuracy is reported together with overall accuracy because the nine tissue classes are not perfectly balanced.

---

## Additional Analyses

The manuscript and supplementary experiments additionally evaluate:

- component ablation;
- branch-control experiments;
- independent topology-module validation;
- leave-one-component-out ablations;
- progressive cumulative ablations;
- foreground-threshold sensitivity;
- grayscale robustness;
- controlled stain perturbation;
- patch-shuffle spatial-structure perturbation;
- feature discriminability;
- attribution / faithfulness analysis;
- runtime and computational profiling;
- external TCGA-UT generalization;
- paired statistical comparisons.

The manuscript and supplement provide the analysis protocols and results. Figure 3 uses five fixed tiles per dataset and computes Grad-CAM for the DFDA classifier at the pre-attention normalization layer of DINOv3 block 39 on both DeepHisto and CRC (one-based numbering, corresponding to `blocks[-2].norm1`). The faithfulness analysis uses the same Grad-CAM target layer and recomputes the PH maps after blurring 5%, 10%, or 20% of the tissue and scores DFDA probabilities. The stain and patch-shuffle analyses use transformed test inputs. The paired comparisons require aligned sample IDs, true labels, classes, and predicted probabilities for each seed from both models.

### Foreground-threshold settings used in the main experiments

The main reported settings are:

```text
DeepHisto: threshold 200
CRC:       threshold 180
```

These values are dataset-specific and correspond to the final preprocessing configuration used for the reported benchmark results.

---

## Computational Considerations

Counts from the full models loaded with the seed-42 checkpoints:

| Model | Total parameters | Trainable parameters |
|---|---:|---:|
| DeepHisto (five classes) | 6,869,929,109 | 66,188,949 |
| CRC (nine classes) | 6,869,930,657 | 66,190,497 |

Supplementary Section 4 (Table S2) reports the computational measurements for this configuration, with approximately 6.870B total parameters and 66.190M trainable parameters.

The large DINOv3 backbone is frozen and adapted with LoRA; however, freezing parameters does not remove the inference-time computational cost of executing the backbone.

### Persistent-homology cost

PH maps can be precomputed and cached.

This is the recommended setting for repeated training or evaluation because:

- topology extraction is deterministic under fixed preprocessing;
- repeated online PH construction adds substantial preprocessing latency;
- cached maps make repeated seed-specific experiments more practical.

When online PH extraction is enabled, its preprocessing time should be considered separately from neural-network inference time.

### Scope

The reported experiments evaluate patch-level histopathology classification.

The repository does not establish:

- whole-slide inference validity;
- clinical diagnostic performance;
- prospective clinical utility;
- deployment-time latency for a complete pathology workflow.

---

## Reproducibility Notes

For a reproducible run, verify all of the following before training:

- [ ] The correct official dataset version has been downloaded.
- [ ] DeepHisto preserves the official patient-wise train/test partition.
- [ ] CRC-VAL-HE-7K is not used for training or checkpoint selection.
- [ ] The internal validation subset is drawn only from the official training data.
- [ ] The correct random seed is used.
- [ ] The correct PH foreground threshold is used.
- [ ] The correct PH input resolution is used.
- [ ] All four TopoFPN levels are present.
- [ ] Image identifiers and topology-cache identifiers are aligned.
- [ ] DINOv3 and ConvNeXtV2 checkpoints correspond to the reported pretrained models.
- [ ] The local DINOv3 source tree contains the TopoFormer-specific `vision_transformer.py` modification.
- [ ] The reported optimizer, learning rate, weight decay, epoch count, and regularization settings are unchanged.
- [ ] The selected checkpoint is based on internal validation balanced accuracy.
- [ ] Final metrics use the designated official test set.
- [ ] The fusion scorer uses the matching seed, checkpoint and four-scale caches.
- [ ] Mean and sample standard deviation are computed across the three reported seeds.

### Random seeds

```text
DeepHisto: 42, 123, 666
CRC:       12, 42, 666
```

### Recommended output organization

A reproducible local experiment directory can be organized as:

```text
experiments/
├── deephisto/
│   ├── seed42/
│   ├── seed123/
│   └── seed666/
└── crc/
    ├── seed12/
    ├── seed42/
    └── seed666/
```

For each seed, retain at least:

```text
training configuration
best validation checkpoint
validation metrics
final test metrics
feature cache
logistic-regression output
software / hardware information
```

---

## Citation

If you use TopoFormer or this repository in academic work, please cite the associated manuscript.

```bibtex
@article{li2026topoformer,
  title   = {TopoFormer: A Multi-Scale Topology-Aware Vision Transformer for H\&E Tumor Classification},
  author  = {Li, Zhibo and Peng, Sijun and Liu, Jian and Rodr{\'i}guez, Alfredo and Chan, Kuan Yoow and S{\'a}mano-S{\'a}nchez, Hugo},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Manuscript}
}
```

> Update the BibTeX entry with the final volume, issue, page range, DOI, and publication status after formal publication.

For an archival public release, it is also recommended to provide a repository-level `CITATION.cff` containing the final manuscript metadata.

---

## Third-Party Resources and Dataset Citations

TopoFormer builds on external software, pretrained models, and public datasets. Users should cite the original resources in addition to the TopoFormer manuscript when those resources materially contribute to their work.

### DINOv3

**Paper**

O. Siméoni *et al.*, **“DINOv3,”** arXiv:2508.10104, 2025.

- Paper: https://arxiv.org/abs/2508.10104
- Code: https://github.com/facebookresearch/dinov3

### ConvNeXt V2

**Paper**

S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie,  
**“ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders,”** 2023.

- Paper: https://arxiv.org/abs/2301.00808
- Code: https://github.com/facebookresearch/ConvNeXt-V2

### GUDHI

Persistent-homology computation in this project uses GUDHI.

- Project: https://gudhi.inria.fr/
- Citation guidance: https://gudhi.inria.fr/python/latest/citation.html

The verified A40 environment contains **GUDHI 3.11.0**. The manuscript lists **3.10.0** for the original PH extraction. Preserve the extraction version with each cache. When preparing a publication, cite the GUDHI reference corresponding to the software version used in the experiment.

### DeepHisto

M. Mittelbronn, A.-C. Hau, S.-Y. Kim, P. V. Nazarov, V. Despotovic, A. Kakoichankava, F. B. K. Borgmann, and G. G. Klamminger,  
**“DeepHisto: Dataset for glioma subtype classification from Whole Slide Images,”** Zenodo, Version 0.1, 2023.

- DOI: https://doi.org/10.5281/zenodo.7941080

### NCT-CRC-HE-100K / CRC-VAL-HE-7K

J. N. Kather, N. Halama, and A. Marx,  
**“100,000 histological images of human colorectal cancer and healthy tissue,”** Zenodo, Version 0.1, 2018.

- DOI: https://doi.org/10.5281/zenodo.1214456

The same Zenodo record contains **CRC-VAL-HE-7K**, the 7,180-patch independent CRC evaluation set used by this project.

### TCGA-UT

D. Komura and S. Ishikawa,  
**“Histology images from uniform tumor regions in TCGA Whole Slide Images (TCGA-UT),”** Zenodo.

- Dataset DOI: https://doi.org/10.5281/zenodo.5889558
- Hugging Face mirror: https://huggingface.co/datasets/dakomura/tcga-ut

The associated publication is:

D. Komura *et al.*,  
**“Universal encoding of pan-cancer histology by deep texture representations,”** *Cell Reports*, 38, 110424, 2022.

- DOI: https://doi.org/10.1016/j.celrep.2022.110424

---

## Acknowledgements

This project builds on the following open research resources:

- **DINOv3** — https://github.com/facebookresearch/dinov3
- **ConvNeXt V2** — https://github.com/facebookresearch/ConvNeXt-V2
- **GUDHI** — https://gudhi.inria.fr/
- **PyTorch** — https://pytorch.org/
- **DeepHisto** — https://doi.org/10.5281/zenodo.7941080
- **NCT-CRC-HE-100K / CRC-VAL-HE-7K** — https://doi.org/10.5281/zenodo.1214456
- **TCGA-UT** — https://doi.org/10.5281/zenodo.5889558

We thank the maintainers and authors of these projects and datasets for making their resources available to the research community.

---

## License

This repository is released for academic research purposes under the terms specified in the repository's `LICENSE` file.

Third-party components are **not automatically relicensed** by the TopoFormer repository. Users must separately comply with the licenses and terms associated with:

- DINOv3 source code and pretrained weights;
- ConvNeXt V2 source code and pretrained weights;
- DeepHisto;
- NCT-CRC-HE-100K / CRC-VAL-HE-7K;
- TCGA-UT;
- GUDHI;
- PyTorch;
- any additional third-party package listed in `requirements.txt`.

Pretrained checkpoints and datasets should be downloaded from their original distribution sources unless redistribution is explicitly permitted.

---

## Research Use and Reporting

When reporting results obtained from this repository, please state at minimum:

1. dataset and exact split;
2. PH threshold and PH input resolution;
3. training seed;
4. pretrained backbone checkpoint;
5. validation/checkpoint-selection protocol;
6. whether PH maps were precomputed or generated online;
7. whether the reported value is a single run or mean ± SD;
8. hardware and key software versions;
9. whether the result is patch-level, slide-level, or external evaluation.

This information is important because the repository contains dataset-specific preprocessing and seed-specific evaluation configurations.

---

## Contact

For questions concerning the implementation, reproduction of the reported experiments, or the associated manuscript, please use the repository issue tracker or the contact information provided in the manuscript.
