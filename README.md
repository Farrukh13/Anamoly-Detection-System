# Anamoly-Detection-System

This project implements an end-to-end anomaly detection pipeline for network intrusion detection using the UNSW-NB15 dataset. The main objective is to detect attack traffic by training models only on normal network behaviour.

# Overview

The project compares four anomaly detection models:

Isolation Forest
One-Class SVM
Local Outlier Factor
PyTorch Autoencoder

Each model is trained on normal traffic and evaluated on a labelled test set containing both normal and attack records.

# Key Features

Normal-only training setup
UNSW-NB15 train/test file discovery
Preprocessing for categorical and numerical features
Multiple anomaly detection models
Deep learning autoencoder using PyTorch
Threshold tuning using validation anomaly scores
Repeated runs for stability analysis
Detailed metrics and visual comparisons
Final model recommendation based on performance trade-offs

# Dataset

This project is designed for the UNSW-NB15 cybersecurity dataset.

Expected label format:

0 = Normal traffic
1 = Attack traffic

The following columns are excluded from training when present:

id, label, attack_cat
The attack_cat column is used only for analysis and visualization, not for model training.

# Methodology

The pipeline follows these steps:

Load training and testing CSV files.
Separate features from labels.
Keep only normal samples from the training set.
Split normal traffic into training and validation sets.
Preprocess categorical and numerical features.
Train anomaly detection models.
Score validation and test samples.
Select thresholds using validation score percentiles.
Evaluate predictions on the labelled test set.
Compare models using metrics, plots, and ranking rules.

# Models Used

Isolation Forest:	Detects anomalies by isolating unusual samples in random trees
One-Class SVM:	Learns a decision boundary around normal data
Local Outlier Factor:	Detects samples that differ from their local neighbourhood
Autoencoder:	Learns to reconstruct normal traffic and flags high reconstruction errors

# Evaluation Metrics
The project evaluates each model using:

Precision
Recall
F1-score
ROC-AUC
Average precision
False positive rate
Specificity
Balanced accuracy
Confusion matrix values
Fit time
Scoring time
Visualizations

The experiment generates:

Class distribution plots
Attack category frequency plots
Threshold sensitivity curves
Runtime comparison charts
Stability boxplots
ROC curves
Precision-recall curves
Anomaly score distributions
Confusion matrices
Attack-category detection heatmaps
Metric correlation heatmaps
Autoencoder training loss curves

# Installation
Install the required dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn torch ipython

# Recommendation Strategy
The default ranking rule is:

Highest F1-score
Highest recall
Lowest false positive rate
This helps balance attack detection with practical false alarm control.

# Summary
This project provides a complete benchmark for network intrusion detection using anomaly detection. By training only on normal traffic, it reflects a realistic cybersecurity scenario where unknown or evolving attacks must be detected without relying on labelled attack.
