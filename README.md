# MAPA-CLIP-FDB

**Metadata-Aware Prompt-Aligned CLIP with Fuzzy Dual-Branch Fusion for Few-Shot Skin Lesion Classification**

Official implementation of the paper *"MAPA-CLIP-FDB: Metadata-Aware Prompt-Aligned CLIP with Fuzzy Dual-Branch Fusion for Few-Shot Skin Lesion Classification."*

**Authors:** Talha Khalid, Ashfaq Ahmad, Iqra Javed, Waseem Iqbal, Syed Baqar Hussain, Amber Sultan
University of Management and Technology (UMT), Lahore, Pakistan

---

## Overview

Few-shot medical image classification is difficult because labeled clinical data are scarce, inter-class visual differences are subtle, and expert annotation is expensive. Most existing approaches also rely on visual features alone and ignore the structured clinical metadata collected alongside every dermoscopic or clinical image.

MAPA-CLIP-FDB is a metadata-aware, prompt-aligned multimodal framework for few-shot skin lesion classification. It combines:

- **Fuzzy dual-branch vision stream.** A frozen CLIP ViT-B/16 encoder and a frozen, ImageNet-pretrained DenseNet-121 backbone (with a trainable projection head) are fused through a learnable Gaussian-membership confidence gate, so each branch is weighted by its own activation-based reliability.
- **Metadata-aware text stream.** CoOp-style learnable prompt tokens on a frozen CLIP text encoder are paired with an autoencoding metadata encoder projected into the CLIP embedding space.
- **Learnable logit fusion.** Image–text and metadata–text logits are merged through a learnable sigmoid gate rather than a fixed sum.
- **Distributional harmony objective.** Cross-modal consistency is enforced by four terms (Sinkhorn optimal transport, the Hilbert–Schmidt Independence Criterion, class-conditional moment matching, and a metadata triplet constraint), trained jointly with a metadata reconstruction loss and a label-smoothed cross-entropy loss, **without any explicit contrastive loss**.

Only about 1.8 M parameters are trained, against roughly 156 M frozen encoder parameters.

## Dataset

Experiments use **PAD-UFES-20**, a public clinical skin lesion dataset of 2,298 images from 1,373 patients in six diagnostic classes: ACK, BCC, MEL, NEV, SCC and SEK. Each image comes with structured patient metadata.

- Download: <https://data.mendeley.com/datasets/zr7vgbcyr2/1>
- A 59-dimensional metadata vector is built from:
  - 14 binary clinical and lifestyle indicators
  - 4 standardized continuous attributes
  - 41 one-hot categorical dimensions
- A stratified 80/20 split with `GLOBAL_SEED = 42` gives 1,838 training images and a fixed 460-image test set.
- Each training episode draws a 5-shot, 6-way support set (30 images) from the training pool.

The dataset is not redistributed with this repository. Download it from the official source and follow its terms of use.

## Results

PAD-UFES-20, 5-shot, 6-way, on the fixed 460-image test set (representative run):

| Metric            | Zero-Shot CLIP | MAPA-CLIP-FDB |
|-------------------|:--------------:|:-------------:|
| Accuracy          | 40.65%         | **61.52%**    |
| Balanced Accuracy | 40.33%         | **60.29%**    |
| Macro-F1          | 26.55%         | **57.73%**    |
| Weighted F1       | —              | **62.80%**    |

### Ablation study

| Model Variant              | Accuracy   | Bal. Acc.  | Macro-F1   |
|----------------------------|:----------:|:----------:|:----------:|
| Zero-Shot CLIP (ViT-B/16)  | 40.65%     | 40.33%     | 26.55%     |
| ViT-only (CoOp + CLIP)     | 37.17%     | 17.35%     | 10.48%     |
| CNN-only (DenseNet-121)    | 30.65%     | 31.60%     | 27.26%     |
| Fuzzy Dual-Branch Img-Text | 41.74%     | 37.64%     | 37.36%     |
| Metadata-Text only         | 53.48%     | 54.12%     | 47.60%     |
| **MAPA-CLIP-FDB (full)**   | **61.52%** | **60.29%** | **57.73%** |

The full model reaches a macro-AUC of 0.832, the highest of the six methods. The learned fusion gate converges to σ(α) = 0.6229, a mild preference for the image–text pathway.

## Requirements

- Python 3.12
- PyTorch with CUDA (developed with PyTorch 2.11 and CUDA 12.8) and torchvision
- OpenAI CLIP, plus its dependencies `ftfy`, `regex` and `tqdm`
- NumPy, pandas, scikit-learn, matplotlib, seaborn, Pillow, psutil

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow psutil
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Experiments were run on an AMD Ryzen 7 5800X CPU, 60 GB RAM and a single NVIDIA GeForce RTX 3080 GPU. A full run, including evaluation and all plots, takes about 20–40 minutes on this hardware.

## Usage

The complete pipeline is in the notebook `MAPA-CLIP-FDB.ipynb`: data loading, model, training, evaluation, ablations and figures.

1. Download PAD-UFES-20.
2. Point the notebook to your data with environment variables, or edit the defaults in the configuration cell:

   ```bash
   export DATA_ROOT=/path/to/PAD-UFES-20          # contains metadata.csv
   export CSV_PATH=$DATA_ROOT/metadata.csv
   export IMG_DIR=$DATA_ROOT/images/all_images
   export OUTPUT_DIR=/path/to/output
   ```

3. Run all cells. The notebook picks the GPU with the most free memory, trains for 200 epochs, and saves the following to `OUTPUT_DIR`:
   - the best checkpoint
   - test-set predictions
   - a performance summary
   - confusion matrices, ROC, precision–recall and calibration plots
   - fuzzy-weight plots and PCA embedding plots

### Main hyperparameters

| Setting | Value |
|---|---|
| CLIP backbone | ViT-B/16 (frozen) |
| CNN backbone | DenseNet-121, ImageNet (frozen; projection head trained) |
| Shots / ways | 5 / 6 |
| Prompt context tokens | 16 |
| Epochs | 200 (cosine annealing) |
| Optimizer | AdamW, weight decay 1×10⁻⁴ |
| Learning rates | prompt 2×10⁻³; DenseNet head and fuzzy gate 1×10⁻⁴; metadata modules 1×10⁻⁴ |
| Loss weights (λ_OT, λ_HSIC, λ_moment, λ_recon, λ_trip) | 0.3, 0.2, 0.15, 0.2, 0.1 |
| Sinkhorn | ε = 0.05, 50 iterations |
| Triplet margin | 1.0 |

The support-set seed (`RUN_SEED`) is drawn per run, so support images differ between runs while the test set stays fixed. The results reported above come from run `82084`.

## Limitations

- Reported results come from a single representative training episode. Variance across support-set samplings is not yet fully characterized; three independently sampled episodes ranged from 57.17% to 65.00% accuracy.
- The continuous-feature scaler and categorical vocabulary are fit on the full dataset before splitting. This affects only normalization statistics; no test labels are exposed.
- The loss weights were set manually, and evaluation uses a single dataset.

## Citation

If you use this code, please cite the accompanying paper:

> T. Khalid, A. Ahmad, I. Javed, W. Iqbal, S. B. Hussain, and A. Sultan, "MAPA-CLIP-FDB: Metadata-Aware Prompt-Aligned CLIP with Fuzzy Dual-Branch Fusion for Few-Shot Skin Lesion Classification," 2026.

## License

The source code is released under the [MIT License](LICENSE). PAD-UFES-20, CLIP and DenseNet-121 pretrained weights are subject to their own licenses and terms of use.

## Contact

Talha Khalid — tk839587@gmail.com