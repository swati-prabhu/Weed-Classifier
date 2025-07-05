# Weed Image Classifier using Self-Supervised Learning

This repository contains the implementation of a self-supervised learning (SSL) model for weed image classification. The model utilizes a Swin Transformer backbone along with Multi-Masked Image Modeling (MMIM) to learn robust representations without relying on labeled data during training.

This marks my **first Machine Learning model** being published on GitHub, and I am excited to share it with the community.

---

## 🔗 Live Demo

You can interact with the trained model directly on **Hugging Face Spaces**:

 [Launch the Model on Hugging Face](https://huggingface.co/spaces/NagashreePai/Weed_Classifier)

---

##  Project Description

Efficient weed classification is crucial for precision agriculture. This model is designed to classify images of various weed species using a self-supervised learning approach. It leverages:
- **Swin Transformer**: A hierarchical vision transformer that enables strong performance with lower computational cost.
- **Multi-Masked Image Modeling (MMIM)**: An SSL strategy that masks multiple image patches to improve feature learning.

---

##  Key Features

-  Transformer-based feature extraction with **Swin Transformer**
-  Self-supervised learning via **MMIM**
-  Trained on a **custom weed image dataset**
-  Implemented in **PyTorch**
-  Live model available on Hugging Face Spaces

---

##  Repository Structure
<pre lang="markdown"> ``` WeedClassifier/ ├── ssl014_vs_code_ready.py # Self-supervised training script ├── weed_test.py # Evaluation and inference script ├── MMIM_checkpoints/ # Directory for model checkpoints │ └── MMIM_best.pth ├── .gitignore ├── .gitattributes ├── README.md # Project documentation ``` </pre>
