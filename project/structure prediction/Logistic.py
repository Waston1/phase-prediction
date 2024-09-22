import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# Load the CSV file
file = 'train_data.csv'
data = pd.read_csv(file)

# Select features and target
y = data['Phases']
X = data.drop(['Alloy ', 'Phases'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Define parameter grid for Logistic Regression
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 0.01, 0.001]
}

# Initialize and fit Logistic Regression with GridSearchCV
model_lr = LogisticRegression(solver='liblinear')
lr_cv = GridSearchCV(estimator=model_lr, param_grid=param_grid, scoring='roc_auc', cv=4)
lr_cv.fit(X_train, y_train)
print('Best Logistic Regression parameters:', lr_cv.best_params_)

# Predict probabilities for each class on the test set
y_score = lr_cv.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(lr_cv.classes_)):
    fpr[i], tpr[i], _ = metrics.roc_curve((y_test == i).astype(int), y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(lr_cv.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()

# Calculate Macro-ROC AUC
macro_auc = np.mean(list(roc_auc.values()))
print('Macro AUC:', macro_auc)

# Confusion matrix
y_pre = lr_cv.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pre, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.0%', cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('Classification Report:')
print(metrics.classification_report(y_test, y_pre, target_names=[f'Class {i}' for i in range(len(lr_cv.classes_))]))

# Feature Importance Scores for Logistic Regression
coefficients = lr_cv.best_estimator_.coef_

# Since Logistic Regression can be multi-class, we handle coefficients for each class
for i, class_label in enumerate(lr_cv.classes_):
    coef = coefficients[i]
    feature_names = X.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coef)})

    # Normalize the feature importance scores
    feature_importances['Normalized Importance'] = feature_importances['Importance'] / feature_importances[
        'Importance'].sum()

    # Sort by importance
    feature_importances = feature_importances.sort_values(by='Normalized Importance', ascending=False)

    # Plot feature importance scores for each class
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Normalized Importance', y='Feature', data=feature_importances, palette='Blues_d')
    plt.title(f'Normalized Feature Importance Scores for Class {class_label}')
    plt.xlabel('Normalized Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
