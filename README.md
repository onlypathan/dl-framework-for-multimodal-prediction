# Gated Multimodal Deep Learning for Restaurant Rating Prediction

**Author:** Rizwan Ahasan Pathan  
**Affiliation:** M.S. in Applied Data Science, University of Southern California (USC)  
**Project Type:** Deep Learning · Multimodal Learning · Applied ML Systems  
**Primary Focus:** Multimodal Fusion, Representation Learning, Model Interpretability  

---

## Overview

This project implements a **gated multimodal deep learning framework** to predict restaurant star ratings by jointly learning from **structured metadata, textual reviews, images, and geographic context**.

Unlike traditional approaches that rely on a single signal (e.g., ratings or review text), the model integrates **four complementary modalities** using domain-specific encoders and a **learnable gated fusion mechanism**. The system dynamically weights each modality based on its informativeness, improving robustness to noisy or missing data.

The model is trained and evaluated on the **Yelp Open Dataset (2024)** and achieves **state-of-the-art regression performance**, demonstrating how multimodal fusion and spatial context significantly enhance real-world prediction accuracy.


<p align="center">
  <img src="images/architec.png" width="80%" />
</p>


<p align="center">
  <img src="images/pie_chart.png" width="48%" />
  <img src="images/final_model_compare.png" width="48%" />
</p>


---

## Problem Motivation

Online restaurant ratings are highly influential but often **biased, subjective, and incomplete**.  
Key challenges include:
- Noisy user reviews
- Missing or sparse images
- Ignoring neighborhood and spatial effects
- Over-reliance on shallow metadata

This project addresses these limitations by designing a **scalable, multimodal regression architecture** that learns a holistic representation of restaurant quality.

---

## Data Modalities

The model integrates the following inputs:

1. **Structured Business Metadata**
   - Review count, price tier, categories, geolocation
   - 670 features → encoded via a multi-layer perceptron (MLP)

2. **Textual Reviews**
   - Raw Yelp reviews (≤ 512 tokens)
   - Encoded using fine-tuned **BERT (bert-base-uncased)**

3. **Restaurant Images**
   - Food and ambience photos
   - Encoded using **ResNet-50** pretrained on ImageNet

4. **Geographic Context**
   - ZIP-code-level spatial relationships
   - Modeled using a **Graph Neural Network (GCN)** over ZIP-code graphs

---

## Model Architecture

Each modality is processed by a dedicated encoder:

| Modality | Encoder | Embedding Dim |
|--------|--------|----------------|
| Metadata | MLP (3 layers) | 256 |
| Text | BERT | 768 |
| Images | ResNet-50 | 2048 |
| Geography | GNN (GCN) | 32 |

A **gated fusion module** learns a scalar weight for each modality, allowing the model to:
- Emphasize informative signals
- Down-weight noisy or missing inputs
- Improve interpretability and stability

The fused representation is passed to a regression head to predict continuous star ratings.

---

## Training & Evaluation

- **Dataset:** Yelp Open Dataset (2024)
  - 36,680 restaurants
  - ~4.3M reviews
  - ~200K images
- **Train/Test Split:** 80/20 (business-level, no leakage)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** AdamW
- **Hyperparameter Tuning:** Optuna (20 trials)
- **Hardware:** NVIDIA Tesla V100 (32GB)

### Final Test Performance

| Metric | Score |
|-----|------|
| RMSE | **0.2654** |
| MAE | **0.2042** |
| R² | **0.8940** |
| Pearson Corr. | **0.9456** |

The gated fusion model significantly outperforms all single-modality baselines.

---

## Key Findings

- **Text reviews** are the strongest single signal, but insufficient alone
- **Images and metadata** add complementary value
- **Geographic context (GNN)** improves differentiation across neighborhoods
- **Gated fusion** is critical for robustness and generalization

Average modality contribution:
- Text (BERT): ~45.8%
- Images (CNN): ~30.3%
- Metadata (MLP): ~19.4%
- Geography (GNN): ~4.4%

---

## Repository Structure


- `data/` – Normalized and scaled datasets stored in a SQLite database (excluded due to GitHub size limits; access can be provided upon request)
- `images/` – Figures and visual results 
- `src/` – Model architecture, modality encoders, gated fusion logic, and training scripts  
  - `raw_image/` – Raw restaurant images from the Yelp Open Dataset (2024), used for CNN-based feature extraction 
- `Final_Report-Deep_Learning.pdf` – Full technical report (Pages 15)  
- `README.md` – Project documentation and usage overview  
- `requirements.txt` – Python dependencies required to run the project  

> **Note:** Due to GitHub size limits, the SQLite database is not included in this repository.
> See `data/README.md` for details on reproducing or requesting the dataset.

---

## Requirements

- Python 3.9 or later  
- PyTorch, Transformers, and scikit-learn for deep learning and modeling  
- Additional libraries for NLP, image processing, graph learning, and visualization  

All required Python packages are listed in `requirements.txt`.

---

INSTALLATION
------------
Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Note:
Installing torch-geometric may require platform-specific wheels.
Refer to the official installation guide:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


---

RUNNING EXPERIMENTS
-------------------
Training scripts for individual modalities (MLP, BERT, CNN, GNN) and the gated fusion
model are located in the src/ directory. Each script can be executed independently
to reproduce the corresponding embeddings or prediction models.


---


## Reproducibility

- Fixed random seeds across all experiments
- Saved checkpoints and embeddings
- Logged loss curves and evaluation metrics
- Modular encoders enable easy ablation or extension

---

## Future Work

- Temporal modeling of reviews and ratings
- Finer-grained spatial graphs (block-level)
- Improved handling of label imbalance
- Deployment as a recommendation or ranking service
- Extension to other domains (hotels, retail, POIs)

---

## References

- [Yelp Open Dataset](https://www.yelp.com/dataset)

- [Liu, Z. (2020). *Yelp Review Rating Prediction: Machine Learning and Deep Learning Models*](https://arxiv.org/abs/2012.06690)

- [Zhao, Y. et al. (2024). *Multimodal Point-of-Interest Recommendation*](https://arxiv.org/abs/2410.03265)

- [Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy*](https://ieeexplore.ieee.org/document/8269806)

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

- [scikit-learn Documentation](https://scikit-learn.org/stable/)

- [Optuna Hyperparameter Optimization](https://optuna.org/)

- [Graph Neural Networks (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/)
