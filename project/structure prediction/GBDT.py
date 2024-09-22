import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file = 'train_data.csv'
data = pd.read_csv(file)

# Select features and target
y = data['Phases']
X = data.drop(['Alloy ', 'Phases'], axis=1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

# Implementing cost-sensitive learning
sample_weight = []
for i in range(len(y_train)):
    if y_train.iloc[i] == 0:
        sample_weight.append(0.2)
    else:
        sample_weight.append(0.8)

# Initialize Gradient Boosting Classifier
clf = ensemble.GradientBoostingClassifier()

# Fit the model on the training data
gbdt_model = clf.fit(X_train, y_train, sample_weight=sample_weight)

# Predict probabilities for each class on the test data
y_score = gbdt_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class, but only for classes present in y_test
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, class_label in enumerate(gbdt_model.classes_):
    if np.sum(y_test == class_label) > 0:  # Ensure the class is present in y_test
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_score[:, i], pos_label=class_label)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    else:
        roc_auc[i] = None  # If the class is not in y_test, skip AUC calculation

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(gbdt_model.classes_):
    if roc_auc[i] is not None:  # Plot only for classes present in y_test
        plt.plot(fpr[i], tpr[i], label=f'Class {class_label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()

# Calculate Macro-ROC AUC
auc_scores = []
for i, class_label in enumerate(gbdt_model.classes_):
    if np.sum(y_test == class_label) > 0:
        auc_score = metrics.roc_auc_score((y_test == class_label).astype(int), y_score[:, i])
        auc_scores.append(auc_score)

macro_auc = np.mean(auc_scores)
print('Macro AUC:', macro_auc)

# Classification Report
y_pred = gbdt_model.predict(X_test)
print('Classification Report:')
print(metrics.classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in gbdt_model.classes_]))

# Plot the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.0%', cmap='Blues', xticklabels=[f'Class {i}' for i in gbdt_model.classes_],
            yticklabels=[f'Class {i}' for i in gbdt_model.classes_])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
