Metabolite Data Analysis

This repository contains code for analyzing a dataset of urinary metabolite data. The analysis includes exploratory data analysis (EDA), preprocessing, model training, evaluation, and comparison between K-Nearest Neighbors (KNN) classifier and Gaussian Naive Bayes classifier.

Dataset

The dataset (urinedata.csv) contains information about urinary metabolite levels. Each row represents a sample, and the columns include metabolite measurements (Ort, Hip, Gly, Val) and the corresponding class label (Name).

Analysis Steps

Data Loading and Exploration:

The data is loaded from the CSV file using pandas.
Exploratory analysis is conducted, including checking data shape, variable distributions, data types, and presence of null values.

Data Visualization:
Histograms are plotted to visualize the distribution of metabolite measurements.

Model Training and Evaluation:

The data is split into training, validation, and test sets.
Features are standardized using StandardScaler.
A KNN classifier is trained and evaluated on the training, validation, and test sets.
A Gaussian Naive Bayes classifier is trained and evaluated on the same sets.
Confusion matrices are plotted to evaluate classification performance.
Accuracy scores are calculated for each classifier on different sets.

Comparison:
A comparison of accuracy scores between KNN and Gaussian Naive Bayes classifiers is visualized using bar plots.
Results

The analysis reveals the performance of KNN and Gaussian Naive Bayes classifiers on the urinary metabolite dataset. Both classifiers are evaluated based on their accuracy scores on the training, validation, and test sets. Confusion matrices provide insights into the classification performance for each class.
