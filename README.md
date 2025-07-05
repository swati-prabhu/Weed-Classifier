# Weed Image Classifier using Self-Supervised Learning

This repository contains the implementation of a self-supervised learning (SSL) model for weed image classification. The model utilizes a Swin Transformer backbone along with Multi-Masked Image Modeling (MMIM) to learn robust representations without relying on labeled data during training.

This marks my **first Machine Learning model** being published on GitHub, and I am excited to share it with the community.

---

## 🔗 Live Demo

You can interact with the trained model directly on **Hugging Face Spaces**:

 [Launch the Model on Hugging Face](https://tinyurl.com/Weed-Classifier)

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

##  Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/swati-prabhu/Weed-Classifier.git
cd Weed-Classifier
---
### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install torch torchvision scikit-learn matplotlib tqdm
---
### 3. Train the Model

Run the training script. You may need to adjust dataset paths inside the script.

```bash
python ssl014_vs_code_ready.py

---

### 4. Evaluate the Model

To test the trained model on new images, run the following script:

```bash
python weed_test.py

---

## Checkpoints

Model checkpoints generated during training are stored in the `MMIM_checkpoints/` directory.

The best-performing model is saved as:


You can load this checkpoint in `weed_test.py` for evaluation or inference.  
Make sure the path to this checkpoint is correctly set in your testing script.

---

## Deployment

This model has been deployed and is available for live testing via Hugging Face Spaces.

You can try out the model here:

 [Hugging Face Space - Weed Classifier](https://huggingface.co/spaces/NagashreePai/Weed_Classifier)

The interface allows you to upload a weed image and get a predicted class based on the trained model.

---

## How to Use & Contribute

This project is intended for anyone interested in self-supervised learning, plant disease detection, or real-world applications of transformers in agriculture.

To get started:
- Clone the repository and install the required dependencies
- Train the model using your dataset or fine-tune the existing model
- Evaluate or test the model using `weed_test.py`
- Optionally, explore or deploy it via the provided Hugging Face Space

Feel free to use this code for your own research, projects, or learning!

### Contributions Welcome

If you find this project helpful or would like to improve it:
- Star the repository
- Fork it and submit a pull request with improvements
- Report bugs or suggest features via [Issues](https://github.com/swati-prabhu/Weed-Classifier/issues)

Your feedback, suggestions, and contributions are greatly appreciated and encouraged!

---

##Contact

I'm excited to share this ML model on GitHub, and I’d love to hear what you think!

If you have questions, suggestions, or just want to connect:
- Open an issue on this repository
- Leave a comment on the [Hugging Face Space](https://tinyurl.com/Weed-Classifier)

Thank you for checking out this project!


