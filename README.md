# Physics-Guided Machine Learning

**Candidate:** Bryan Chen (M.Sc. Applied Mathematics & Computer Science, École Polytechnique)  
**Target Project:** Physics Guided Machine Learning on Real Lensing Images  
**Organization:** ML4Sci / DeepLense

## Overview

This repository contains my solutions for the **DeepLense Google Summer of Code (GSoC)** evaluation tasks. I have completed the full suite of Common and Specific tests to demonstrate my technical capabilities and readiness to contribute to the *Physics Guided Machine Learning* project (either for independent research or GSoC 2026).

The work focuses on applying Deep Learning to strong gravitational lensing data, ranging from standard CNN baselines to advanced **Physics-Informed Neural Networks (PINNs)**, **Diffusion Models**, and **Foundation Models**.

## Repository Structure

The solutions are organized as self-contained Jupyter Notebooks in the `notebooks/` directory:

```text
└── notebooks/
    ├── multiclass_classification.ipynb  # Common Test I: ResNet Baseline & Grad-CAM
    ├── lens-finding.ipynb               # Specific Test II: Binary Classification & Imbalance Handling
    ├── image-super-resolution.ipynb     # Specific Test III: SRResNet & Spectral Analysis
    ├── diffusion-model.ipynb            # Specific Test IV: DDPM Generative Model
    ├── physics-guided-ml.ipynb          # Specific Test V: PINN with Differentiable Ray-Tracing
    └── foundation-model.ipynb           # Specific Test VI: Masked Autoencoder (MAE) for Classif. & SR
```

---

## Summary of Results

### 1. Physics-Guided Machine Learning (Test V) - *Primary Interest*
**Objective:** Integrate the gravitational lensing equation directly into a neural network to regularize training and improve interpretability.
*   **Methodology:** Implemented a **PINN** using a ResNet backbone with a custom differentiable physics layer (Singular Isothermal Sphere model). I utilized **Curriculum Learning**, gradually increasing the weight of the physics loss ($\lambda_{physics}$) from 0 to 1.0 during training.
*   **Key Results:**
    *   Achieved **~95% Validation Accuracy**.
    *   Successfully reconstructed the un-lensed source plane during the forward pass.
    *   Extracted physical parameters (Einstein Radius $\theta_E$) distributions that correlate physically with the substructure classes.

### 2. Multi-Class Classification (Common Test I)
**Objective:** Establish a strong baseline for categorizing lensing substructures (`no`, `sphere`, `vortex`).
*   **Methodology:** Fine-tuned ResNet-18. Performed extensive error analysis using **Grad-CAM**.
*   **Key Results:**
    *   **93.69% Accuracy**, **0.9907 AUC**.
    *   **Analysis:** Grad-CAM revealed that standard CNNs often "cheat" by focusing on high-luminosity artifacts rather than physical distortions, providing the motivation for the Physics-Guided approach in Test V.

### 3. Lens Finding (Test II)
**Objective:** Identify lenses in a highly imbalanced dataset (~100:1 non-lens to lens ratio).
*   **Methodology:** Implemented a rigorous pipeline with **Weighted BCE Loss**, aggressive data augmentation, and F1-score threshold tuning.
*   **Key Results:**
    *   **Test AUC: 0.9862**.
    *   Achieved high recall (>80%) on the minority class while maintaining precision, solving the imbalance challenge without over-rejecting candidates.

### 4. Image Super-Resolution (Test III)
**Objective:** Upscale low-resolution lensing images while preserving scientific fidelity.
*   **Methodology:** Developed an **SRResNet** trained with L1 Loss. Evaluation went beyond PSNR/SSIM to include **FFT Power Spectrum Analysis**.
*   **Key Results:**
    *   **PSNR: 42.35 dB**, **SSIM: 0.979**.
    *   **Scientific Validity:** Spectral analysis confirmed the recovery of high-frequency spatial components (dark matter signatures) that bicubic interpolation smoothed out.

### 5. Generative Diffusion Models (Test IV)
**Objective:** Simulate realistic strong lensing images.
*   **Methodology:** Built a **Denoising Diffusion Probabilistic Model (DDPM)** with a U-Net backbone using sinusoidal time embeddings.
*   **Key Results:**
    *   **FID Score:** Reduced from ~239 to **16.60**.
    *   **Latent Walk:** Spherical Linear Interpolation (Slerp) showed smooth transitions between lensing configurations, proving the model learned the continuous physical manifold of the data.

### 6. Foundation Model (Test VI)
**Objective:** Leverage Self-Supervised Learning (SSL) for downstream tasks.
*   **Methodology:**
    *   **Pre-training:** Trained a **Masked Autoencoder (MAE)** on unlabeled data (75% masking ratio).
    *   **Task A (Classification):** Fine-tuned encoder achieving **99.34% Accuracy** (SOTA), outperforming the supervised baseline.
    *   **Task B (Super-Res):** Adapted the encoder with a pixel-shuffle head and **Zero-Initialization** strategy, achieving **43.67 dB PSNR** (+1.54 dB over baseline).

---

## Technologies Used
*   **Core:** Python, PyTorch, NumPy, Pandas.
*   **Vision:** `torchvision`, `timm`, `opencv`, `albumentations`.
*   **Analysis:** `scikit-learn` (Metrics, t-SNE), `scipy` (FFT), `matplotlib`, `seaborn`.
*   **Interpretability:** `pytorch-grad-cam`.

## Usage
Each notebook is self-contained. To reproduce the results:
1.  Clone the repository.
2.  Download the respective datasets from the links provided in the DeepLense task description.
3.  Update the `Config` class in the notebook with your local path to the dataset.
4.  Run the notebook cells sequentially.

## Contact
**Bryan Chen**  
**École Polytechnique**
*Email:* bryan.chen@polytechnique.edu
