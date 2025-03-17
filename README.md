# This repository provides a comprehensive pipeline for analyzing multi-omics data using Flexynesis, focusing on survival analysis, classification, and multi-task modeling. The analysis utilizes mutational (mut) and copy number alteration (cna) data to predict clinical outcomes.

## Notebook Overview

## Data Loading & Preprocessing

Download and extract the dataset (lgggbm_tcga_pub_processed).
Inspect data structure and handle missing values.
# Survival Analysis

Generate Kaplan-Meier survival curves.
Apply Cox Proportional Hazards Model.
Predict survival using Flexynesis.
# Classification & Multi-Task Learning

# Train classification models for HISTOLOGICAL_DIAGNOSIS.
Implement multi-task learning for diagnosis and performance scores.
Create confusion matrices and evaluate metrics.
# Technical Requirements

Python: 3.11+
Environment: Jupyter Notebook / Terminal
Required Libraries:
flexynesis
pandas
numpy
seaborn
matplotlib
scikit-learn
lifelines

# 📊 Visualization outcomes

Confusion Matrix & Classification Report.
Survival Curve (Kaplan-Meier Plot).
