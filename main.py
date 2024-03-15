import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load your data from the CSV file
file_path = 'urinedata.csv'
df = pd.read_csv(file_path)

print('The shape of the data set: ', df.shape)
# Convert numerical columns to float
df[['Ort', 'Hip', 'Gly', 'Val']] = df[['Ort', 'Hip', 'Gly', 'Val']].astype(float)

# Split the data into features (X) and target (y)
X = df[['Ort', 'Hip', 'Gly', 'Val']]
y = df['Name']

for var in df.columns:
    print('Distribution of variables: ', df[var].value_counts())

print('The types of variables', df.dtypes)

print('Check the existence of null values: ', df.isnull().sum())

print('The number of entries for each class: ', df['Name'].value_counts())

print('The percentage of entries for each class: ', df['Name'].value_counts()/float(len(df)))

df.hist(bins=20, figsize=(10, 6))
plt.suptitle("Histograms for Each Metabolite", y=0.95)
plt.show()


grouped_data = df.groupby('Name')

# Plot histograms for each column, organized by 'Name' class
for name, group in grouped_data:
    group.hist(bins=20, figsize=(10, 6))
    plt.suptitle(f"Histograms for Each Metabolite - {name}", y=0.95)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

correlation = df.corr()
correlation['Name'].sort_values(ascending=False)
print('Check how each attribute correlates with the Name variable:', correlation['Name'].sort_values(ascending=False))

# Split the data into training (70%), validation (20%), and testing (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, shuffle=True, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
k = 5  # You can adjust the number of neighbors as needed
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train_scaled, y_train)

y_train_pred = knn_classifier.predict(X_train_scaled)

# Evaluate the accuracy on the training set
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy_train * 100:.2f}%")

# Make predictions on the validation set
y_val_pred = knn_classifier.predict(X_val_scaled)

# Evaluate the accuracy on the validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")

# Make predictions on the test set
y_test_pred = knn_classifier.predict(X_test_scaled)

# Evaluate the accuracy on the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

print('Test number of entries', y_test.value_counts())

null_accuracy = (66/(116))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

cm = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Validation Set')
plt.show()

cm = confusion_matrix(y_train, y_train_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Train Set')
plt.show()

nb_classifier = GaussianNB()

# Train the Gaussian Naive Bayes classifier
nb_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set using Gaussian Naive Bayes
y_test_pred_nb = nb_classifier.predict(X_test_scaled)


y_train_pred_nb = nb_classifier.predict(X_train_scaled)

# Evaluate the accuracy on the training set using Gaussian Naive Bayes
accuracy_train_nb = accuracy_score(y_train, y_train_pred_nb)
print(f"Gaussian NB Training Accuracy: {accuracy_train_nb * 100:.2f}%")

# Make predictions on the validation set using Gaussian Naive Bayes
y_val_pred_nb = nb_classifier.predict(X_val_scaled)

# Evaluate the accuracy on the validation set using Gaussian Naive Bayes
accuracy_val_nb = accuracy_score(y_val, y_val_pred_nb)
print(f"Gaussian NB Validation Accuracy: {accuracy_val_nb * 100:.2f}%")

# Evaluate the accuracy on the test set using Gaussian Naive Bayes
accuracy_test_nb = accuracy_score(y_test, y_test_pred_nb)
print(f"Gaussian NB Test Accuracy: {accuracy_test_nb * 100:.2f}%")

# Create confusion matrix for Gaussian Naive Bayes
cm_nb = confusion_matrix(y_test, y_test_pred_nb)

# Display confusion matrix for Gaussian Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Gaussian NB)')
plt.show()

# Create confusion matrix for Gaussian Naive Bayes
cm_nb = confusion_matrix(y_train, y_train_pred_nb)

# Display confusion matrix for Gaussian Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Train Set(Gaussian NB)')
plt.show()

# Create confusion matrix for Gaussian Naive Bayes
cm_nb = confusion_matrix(y_val, y_val_pred_nb)

# Display confusion matrix for Gaussian Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Validation Set (Gaussian NB)')
plt.show()

classifiers = ['KNN', 'Gaussian NB']
train_accuracies = [accuracy_train, accuracy_train_nb]
val_accuracies = [accuracy_val, accuracy_val_nb]
test_accuracies = [accuracy_test, accuracy_test_nb]


bar_width = 0.2

index = np.arange(len(classifiers))

plt.bar(index, train_accuracies, bar_width, label='Train', color='skyblue')
plt.bar(index + bar_width, val_accuracies, bar_width, label='Validation', color='coral')
plt.bar(index + 2 * bar_width, test_accuracies, bar_width, label='Test', color='limegreen')

plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy: KNN vs Gaussian NB')
plt.xticks(index + bar_width, classifiers)
plt.legend()
plt.show()