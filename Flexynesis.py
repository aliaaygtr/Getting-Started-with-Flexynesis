#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create a new environment with Python 3.11
get_ipython().system('mamba create --name flexynesisenv python=3.11 -y')

# Activate the environment
get_ipython().system('mamba activate flexynesisenv')

# Install Flexynesis
get_ipython().system('python -m pip install flexynesis --upgrade')


# In[2]:


# Download the dataset
get_ipython().system('wget -O lgggbm_tcga_pub_processed.tgz https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/lgggbm_tcga_pub_processed.tgz')

# Extract the dataset
get_ipython().system('tar -xzvf lgggbm_tcga_pub_processed.tgz')


# In[3]:


import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# List the structure of the extracted dataset
list_files("lgggbm_tcga_pub_processed")


# In[4]:


# Run Flexynesis for regression
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --target_variables KARNOFSKY_PERFORMANCE_SCORE              --data_types mut,cna              --hpo_iter 1')


# In[7]:


import pandas as pd

# Load and subsample the dataset
mut_data = pd.read_csv("lgggbm_tcga_pub_processed/train/mut.csv").sample(frac=0.5, random_state=42)
cna_data = pd.read_csv("lgggbm_tcga_pub_processed/train/cna.csv").sample(frac=0.5, random_state=42)
clin_data = pd.read_csv("lgggbm_tcga_pub_processed/train/clin.csv").sample(frac=0.5, random_state=42)

# Save the subsampled data
mut_data.to_csv("lgggbm_tcga_pub_processed/train/mut_subsampled.csv", index=False)
cna_data.to_csv("lgggbm_tcga_pub_processed/train/cna_subsampled.csv", index=False)
clin_data.to_csv("lgggbm_tcga_pub_processed/train/clin_subsampled.csv", index=False)


# In[8]:


import pandas as pd

# Create a sample predictions DataFrame
data = {
    "true": [1.2, 2.3, 3.4, 4.5, 5.6],
    "predicted": [1.1, 2.2, 3.3, 4.4, 5.5]
}

predictions = pd.DataFrame(data)

# Save the DataFrame to a CSV file
predictions.to_csv("predictions.csv", index=False)

# Load the CSV file
predictions = pd.read_csv("predictions.csv")

# Scatter plot of true vs predicted values
plt.scatter(predictions["true"], predictions["predicted"], s=10)
plt.plot([min(predictions["true"]), max(predictions["true"])], [min(predictions["true"]), max(predictions["true"])], color="red", linestyle="--")
plt.title("True vs Predicted Values (Regression)")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()


# In[9]:


# Run Flexynesis for classification
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --target_variables HISTOLOGICAL_DIAGNOSIS              --data_types mut,cna              --hpo_iter 1')


# In[12]:


# Create a sample predictions DataFrame for classification
data = {
    "true": [0, 1, 0, 1, 0, 1, 0, 1],
    "predicted": [0, 1, 0, 0, 1, 1, 0, 1]
}

predictions = pd.DataFrame(data)

# Compute confusion matrix
cm = confusion_matrix(predictions["true"], predictions["predicted"])

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Classification)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[13]:


# Run Flexynesis for survival analysis
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --surv_event_var OS_STATUS              --surv_time_var OS_MONTHS              --data_types mut,cna              --hpo_iter 1')


# In[ ]:


# Run Flexynesis for multi-task learning
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --target_variables HISTOLOGICAL_DIAGNOSIS,KARNOFSKY_PERFORMANCE_SCORE              --surv_event_var OS_STATUS              --surv_time_var OS_MONTHS              --data_types mut,cna              --hpo_iter 1')


# In[ ]:


# Run Flexynesis for unsupervised training
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class supervised_vae              --data_types mut,cna              --hpo_iter 1')


# In[17]:


# Run Flexynesis for cross-modality training
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class CrossModalPred              --data_types mut,cna              --input_layers mut              --output_layers cna              --hpo_iter 1')


# In[18]:


import pandas as pd
import numpy as np

# Generate synthetic survival data
np.random.seed(42)
n_samples = 100
survival_data = pd.DataFrame({
    "OS_MONTHS": np.random.exponential(scale=12, size=n_samples),  # Survival time in months
    "OS_STATUS": np.random.randint(0, 2, size=n_samples)  # Event indicator (0 = censored, 1 = event)
})

# Save the data to a CSV file
survival_data.to_csv("survival_predictions.csv", index=False)


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load survival data
survival_data = pd.read_csv("survival_predictions.csv")

# Fit Kaplan-Meier curve
kmf = KaplanMeierFitter()
kmf.fit(survival_data["OS_MONTHS"], survival_data["OS_STATUS"], label="Survival Curve")

# Plot the survival curve
kmf.plot()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.show()


# In[21]:


import pandas as pd
import numpy as np

# Generate synthetic embeddings
np.random.seed(42)
n_samples = 100
embeddings = pd.DataFrame({
    "embedding_1": np.random.normal(loc=0, scale=1, size=n_samples),
    "embedding_2": np.random.normal(loc=0, scale=1, size=n_samples)
})

# Save the embeddings to a CSV file
embeddings.to_csv("embeddings.csv", index=False)


# In[22]:


import pandas as pd
import umap
import matplotlib.pyplot as plt

# Load embeddings
embeddings = pd.read_csv("embeddings.csv")

# Reduce dimensionality using UMAP
reducer = umap.UMAP()
umap_embeddings = reducer.fit_transform(embeddings)

# Plot the embeddings
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5)
plt.title("UMAP Visualization of Embeddings (Unsupervised)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


# In[23]:


import pandas as pd
import numpy as np

# Generate synthetic cross-modality data
np.random.seed(42)
n_samples = 100
cross_modality_data = pd.DataFrame({
    "mut_input": np.random.normal(loc=0, scale=1, size=n_samples),  # Input modality (e.g., mutation data)
    "cna_output": np.random.normal(loc=0, scale=1, size=n_samples)  # Output modality (e.g., CNA data)
})

# Save the data to a CSV file
cross_modality_data.to_csv("cross_modality_predictions.csv", index=False)


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

# Load cross-modality predictions
cross_modality_data = pd.read_csv("cross_modality_predictions.csv")

# Scatter plot of input vs output values
plt.scatter(cross_modality_data["mut_input"], cross_modality_data["cna_output"], s=10)
plt.title("Cross-Modality Predictions")
plt.xlabel("Input Modality (Mutation)")
plt.ylabel("Output Modality (CNA)")
plt.show()


# In[25]:


# Run Flexynesis with fine-tuning
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --data_types mut,cna              --target_variables HISTOLOGICAL_DIAGNOSIS              --finetuning_samples 50              --hpo_iter 1')


# In[27]:


# Run Flexynesis with feature filtering
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --data_types mut,cna              --target_variables HISTOLOGICAL_DIAGNOSIS              --variance_threshold 1              --features_top_percentile 20              --correlation_threshold 0.8              --hpo_iter 1')


# In[ ]:


# Run Flexynesis with custom HPO settings
get_ipython().system('flexynesis --data_path lgggbm_tcga_pub_processed              --model_class DirectPred              --data_types mut,cna              --target_variables HISTOLOGICAL_DIAGNOSIS              --hpo_iter 50              --hpo_patience 20')


# In[ ]:




