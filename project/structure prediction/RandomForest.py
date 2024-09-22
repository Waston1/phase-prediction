import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

# Load the CSV file
file = 'train_data.csv'
data = pd.read_csv(file)

# Select features and target
y = data['Phases']
X = data.drop(['Alloy ', 'Phases'], axis=1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

# Define parameter grid for Random Forest
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [5, 6, 7, 8],
    'n_estimators': [11, 13, 15, 17],
    'max_features': [0.3, 0.4, 0.5],
    'min_samples_split': [2, 4, 8, 12, 16]
}

# Initialize Random Forest Classifier and perform grid search
rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
print(rfc_cv.best_params_, rfc_cv.best_score_)

# Feature Importance Analysis
importances = rfc_cv.best_estimator_.feature_importances_
feature_names = X.columns

# Create a DataFrame for the feature importances
feat_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importances, palette='Blues_d')
plt.title('Feature Importance Ranking')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Predict probabilities for each class on the test data
y_score = rfc_cv.predict_proba(X_test)

# Compute ROC curve and AUC for each class, but only for classes present in y_test
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, class_label in enumerate(rfc_cv.classes_):
    if np.sum(y_test == class_label) > 0:  # Ensure the class is present in y_test
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_score[:, i], pos_label=class_label)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    else:
        roc_auc[i] = None  # Skip AUC calculation if the class is not present

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(rfc_cv.classes_):
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
for i, class_label in enumerate(rfc_cv.classes_):
    if np.sum(y_test == class_label) > 0:
        auc_score = metrics.roc_auc_score((y_test == class_label).astype(int), y_score[:, i])
        auc_scores.append(auc_score)

if auc_scores:
    macro_auc = np.mean(auc_scores)
    print('Macro AUC:', macro_auc)
else:
    print('No valid AUC scores to compute Macro AUC.')

# Classification Report
y_pred = rfc_cv.predict(X_test)
print('Classification Report:')
print(metrics.classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in rfc_cv.classes_]))

# Plot the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.0%', cmap='Blues',
            xticklabels=[f'Class {i}' for i in rfc_cv.classes_],
            yticklabels=[f'Class {i}' for i in rfc_cv.classes_])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



# 使用训练好的模型创建 SHAP 解释器
explainer = shap.TreeExplainer(rfc_cv.best_estimator_)

# 计算测试集的 SHAP 值
shap_values = explainer.shap_values(X_test)

# 输出 shap_values 数组大小 (注意：每个类别的 SHAP 值都在其中)
print(f"SHAP values shape: {np.array(shap_values).shape}")

# 绘制 SHAP 全局特征重要性图 (针对每个类单独绘制)
for i, class_label in enumerate(rfc_cv.classes_):
    plt.title(f'SHAP Summary Plot for Class {class_label}')
    shap.summary_plot(shap_values[i], X_test, plot_type="dot",show=True)
    plt.show()

