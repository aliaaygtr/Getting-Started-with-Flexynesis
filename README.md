# Flexynesis Multi-Omics Analysis
📌 Notebook Overview
This repository contains code for running Flexynesis on multi-omics benchmark datasets, including survival analysis, classification, and multi-task modeling. The analysis is focused on leveraging mutational (mut) and copy number alteration (cna) data to predict clinical outcomes.

📂 Data Processing & Model Execution
1️⃣ Data Loading & Preprocessing

Download and extract dataset ([lgggbm_tcga_pub_processed](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dataset1.tgz))
Check data structure and clean missing values
2️⃣ Survival Analysis

Kaplan-Meier survival curves
Cox Proportional Hazards Model
Survival prediction using Flexynesis
3️⃣ Classification & Multi-Task Learning

Train classification models for HISTOLOGICAL_DIAGNOSIS
Multi-task learning for diagnosis & performance score
Generate confusion matrices & evaluation metrics

⚙️ Technical Requirements
🛠 Python: 3.11+
🖥 Environment: Jupyter Notebook / Terminal
📦 Required Libraries:
pip install flexynesis pandas numpy seaborn matplotlib scikit-learn lifelines

📈 Expected Outputs
✔️ Kaplan-Meier survival curves
✔️ Cox regression results for survival prediction
✔️ Confusion matrix for classification evaluation
✔️ Training and validation loss curves
✔️ Feature importance ranking for multi-omics prediction

📊 Visualization Tools
 Confusion Matrix & Classification Report
Survival Curve (Kaplan-Meier Plot)
