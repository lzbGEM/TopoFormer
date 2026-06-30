# TopoFormer

TopoFormer is a multi-scale topology-aware visual transformer framework for H&E tumor classification. It integrates persistent homology-based topological representations with DINOv3 visual features, topology-conditioned token interaction, LoRA-based backbone adaptation, feature disentanglement, and ConvNeXt V2 gated feature fusion.

## Overview

TopoFormer is designed for histopathological image classification from H&E-stained tissue images. The framework combines high-capacity visual representation learning with explicit topological descriptors to improve tissue pattern recognition.

The current implementation includes:

- **DINOv3 ViT backbone** for visual representation learning.
- **TopoFPN** for multi-scale topological map construction.
- **TopoCTI** for topology-conditioned token interaction inside the DINOv3 transformer.
- **LoRA adaptation** for efficient DINOv3 fine-tuning.
- **Feature disentanglement** for task-relevant and task-irrelevant representation learning.
- **ConvNeXt V2 gated fusion** for complementary convolutional visual feature integration.
- **Dataset-specific scripts** for DeepHisto and CRC histopathology classification.

## Repository Structure

```text
TopoFormer/
├── Backbone/
│   └── dinov3/
│       └── dinov3/
│           └── dinov3/
│               └── models/
│                   └── vision_transformer.py
├── notebooks/
│   ├── Final_feature_fusion_eval_briancancer.ipynb
│   ├── Final_feature_fusion_eval_CRC.ipynb
│   ├── TopoFPN_layer.ipynb
│   └── TopoFPN_vector.ipynb
├── scripts/
│   ├── train_briancancer_rgb.py
│   └── train_crc_rgb.py
├── topoformer/
│   ├── data/
│   ├── engine/
│   └── model/
├── requirements.txt
└── README.md
```

## Main Entry Points

The main training scripts are located in:

```text
scripts/train_crc_rgb.py
scripts/train_briancancer_rgb.py
```

These scripts are the primary entry points for training, evaluation, and feature cache generation.

## DINOv3 Backbone Modification

This repository includes a local DINOv3 project directory under:

```text
Backbone/dinov3/
```

The included DINOv3 code is adapted from the official DINOv3 implementation. In this project, only the following DINOv3 source file was modified:

```text
Backbone/dinov3/dinov3/dinov3/models/vision_transformer.py
```

This modification enables topology-conditioned token interaction by allowing DINOv3 forward features to receive and inject multi-scale topological maps through the TopoCTI hook.

Original DINOv3 repository:

```text
https://github.com/facebookresearch/dinov3
```

## Installation

Create and activate a Python environment:

```bash
conda create -n topoformer python=3.10 -y
conda activate topoformer
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file is a cleaned project-level dependency file for the current TopoFormer codebase. It is not a full conda environment export.

For CUDA training, please install PyTorch and torchvision according to your CUDA version if the default pip wheels do not match your machine.

## Checkpoint Download

### DINOv3

The DINOv3 checkpoint used in this project is:

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

Download source:

```text
https://github.com/facebookresearch/dinov3
```

Then update the checkpoint path in the training scripts:

```python
DINO_LOCAL_CKPT = "/path/to/your/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
```

### ConvNeXt V2

TopoFormer uses ConvNeXt V2 features for gated fusion. Please download the ConvNeXt V2 checkpoint from:

```text
https://github.com/facebookresearch/ConvNeXt-V2
```

Then update the checkpoint path in the training scripts:

```python
CONVNEXT_CKPT = "/path/to/your/convnextv2_base_1k_224_ema.pt"
CONVNEXT_NAME = "convnextv2_base"
```

## Datasets

| Dataset | Usage | Link |
| --- | --- | --- |
| DeepHisto | Glioma histopathology classification | https://zenodo.org/records/7941080 |
| NCT-CRC-HE-100K / CRC-VAL-HE-7K | Colorectal cancer histology classification | https://zenodo.org/records/1214456 |
| External validation dataset | External glioma validation | To be released / added |

### External Validation Sources

The external validation images were collected from public educational pathology resources. The following sources should be cited or acknowledged when using the external validation setting:

```text
Neuropathology Atlas
https://neuropathologyatlas.wustl.edu/

PEIR Digital Library
https://peir.path.uab.edu/library/

WebPathology - Neuropathology / Glial Tumors
https://www.webpathology.com/images/neuropath/glial-tumors
```

## Data Preparation

### DeepHisto

Download and organize the DeepHisto dataset into training and testing folders.

Example structure:

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

Update the following variables in:

```text
scripts/train_briancancer_rgb.py
```

```python
BRIANCANCER_TRAIN_DIR = "/path/to/your/Briancancer train data"
BRIANCANCER_TEST_DIR = "/path/to/your/Briancancer test data"
```

### CRC

Download the CRC dataset and organize it as:

```text
/path/to/CRC/
├── NCT-CRC-HE-100K/
└── CRC-VAL-HE-7K/
```

Update the following variables in:

```text
scripts/train_crc_rgb.py
```

```python
NCTCRC100K_DIR = "/path/to/your/NCT-CRC-HE-100K"
CRCVAL7K_DIR = "/path/to/your/CRC-VAL-HE-7K"
```

## Topological Feature Preparation

TopoFormer expects precomputed multi-scale topological maps saved as `.npz` files. The training scripts expect four TopoFPN levels:

```text
/path/to/your/topofpn_levels_train_test_maps_L0.npz
/path/to/your/topofpn_levels_train_test_maps_L1.npz
/path/to/your/topofpn_levels_train_test_maps_L2.npz
/path/to/your/topofpn_levels_train_test_maps_L3.npz
```

Set the base path in the training scripts:

```python
TOPO_NPZ_BASE = "/path/to/your/topofpn_levels_train_test_maps"
```

The following notebooks provide examples for TopoFPN construction and evaluation:

```text
notebooks/TopoFPN_layer.ipynb
notebooks/TopoFPN_vector.ipynb
```

## Training

Before training, edit the dataset paths, topological map paths, and checkpoint paths at the beginning of the corresponding script.

### Train on CRC

```bash
python scripts/train_crc_rgb.py
```

Main variables to configure:

```python
NCTCRC100K_DIR = "/path/to/your/NCT-CRC-HE-100K"
CRCVAL7K_DIR = "/path/to/your/CRC-VAL-HE-7K"
TOPO_NPZ_BASE = "/path/to/your/topofpn_levels_train_test_maps"
DINO_LOCAL_CKPT = "/path/to/your/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
CONVNEXT_CKPT = "/path/to/your/convnextv2_base_1k_224_ema.pt"
```

### Train on DeepHisto

```bash
python scripts/train_briancancer_rgb.py
```

Main variables to configure:

```python
BRIANCANCER_TRAIN_DIR = "/path/to/your/Briancancer train data"
BRIANCANCER_TEST_DIR = "/path/to/your/Briancancer test data"
TOPO_NPZ_BASE = "/path/to/your/topofpn_levels_train_test_maps"
DINO_LOCAL_CKPT = "/path/to/your/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
CONVNEXT_CKPT = "/path/to/your/convnextv2_base_1k_224_ema.pt"
```

## Evaluation Notebooks

The repository includes evaluation notebooks for feature fusion and dataset-specific analysis:

```text
notebooks/Final_feature_fusion_eval_briancancer.ipynb
notebooks/Final_feature_fusion_eval_CRC.ipynb
```

## Core Modules

### TopoCTI

TopoCTI injects multi-scale topological maps into the DINOv3 transformer. The implementation is located in:

```text
topoformer/model/topo_cti_v2.py
```

The current stage-level injection follows:

```text
Stage 1 <- L0
Stage 2 <- L1
Stage 3 <- L2
Stage 4 <- L3
```

### TopoFPN

TopoFPN organizes persistent homology-based topological descriptors into multi-scale spatial maps.

Relevant files:

```text
topoformer/data/topo_maps.py
notebooks/TopoFPN_layer.ipynb
notebooks/TopoFPN_vector.ipynb
```

### LoRA Adaptation

LoRA modules are used for parameter-efficient adaptation of the DINOv3 backbone.

```text
topoformer/model/lora.py
```

### Feature Disentanglement

The disentanglement module separates visual representations into task-relevant and task-irrelevant components.

```text
topoformer/model/disentanglement.py
```

### ConvNeXt V2 Gated Fusion

ConvNeXt V2 features are fused with DINOv3 features through a gated fusion module.

```text
topoformer/model/convnext_fusion.py
```

## Reproducibility Notes

- Set all dataset paths and checkpoint paths before running the scripts.
- Ensure the DINOv3 local repository is correctly located under `Backbone/dinov3/`.
- Ensure all four TopoFPN `.npz` files are aligned with the image paths used by the training scripts.
- Large files such as datasets, pretrained checkpoints, cached features, logs, and topological feature archives are recommended to be stored externally rather than directly in the GitHub repository.

## Acknowledgements

This project builds upon the following resources:

- DINOv3: https://github.com/facebookresearch/dinov3
- ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2
- DeepHisto: https://zenodo.org/records/7941080
- NCT-CRC-HE-100K / CRC-VAL-HE-7K: https://zenodo.org/records/1214456
- Neuropathology Atlas: https://neuropathologyatlas.wustl.edu/
- PEIR Digital Library: https://peir.path.uab.edu/library/
- WebPathology: https://www.webpathology.com/images/neuropath/glial-tumors
- GUDHI: https://gudhi.inria.fr/
- PyTorch: https://pytorch.org/

## Citation

If you find this repository useful, please cite the associated manuscript:

```bibtex
@article{li2026topoformer,
  title   = {TopoFormer: A Multi-Scale Topology-Aware Visual Transformer for H\&E Tumor Classification},
  author  = {Li, Zhibo and Peng, Sijun and Jian, Liu and Rodr\'{i}guez, Alfredo and Chan, Kuan Yoow and S\'{a}mano-S\'{a}nchez, Hugo},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

This repository is released for academic research purposes. Please also follow the licenses and usage terms of all third-party codebases, pretrained checkpoints, and datasets used in this project, including DINOv3, ConvNeXt V2, DeepHisto, CRC, and the external validation image sources.
