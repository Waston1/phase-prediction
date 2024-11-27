import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv(r'./IM_SS_AM1.csv')
le = LabelEncoder()
df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results = {}


### 1. Random Forest Model
def train_rf(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(max_depth=13, n_estimators=136, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['Random Forest'] = acc
    print(f"Random Forest Accuracy: {acc}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # SHAP Analysis for RF
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    return rf_model, shap_values


### 2. Support Vector Machine Model
def train_svm(X_train_scaled, X_test_scaled, y_train, y_test):
    svm_model = SVC(C=57.6355, gamma=0.0297, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results['SVM'] = acc
    print(f"SVM Accuracy: {acc}")
                                                                               
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # SHAP Analysis for SVM
    explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_scaled[:100])  # Subset for efficiency
    shap_values = explainer.shap_values(X_test_scaled[:100])
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns)

    return svm_model, shap_values


### 3. Deep Learning Model
def train_deep_learning(X_train_scaled, X_test_scaled, y_train, y_test):
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    model = Sequential([
        Dense(42, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.1353),
        Dense(42, activation='relu'),
        Dropout(0.1353),
        Dense(len(le.classes_), activation='softmax')  # Adjust output for number of classes
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0028),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_scaled, y_train_onehot, epochs=250, batch_size=32, verbose=0, validation_split=0.2,
              callbacks=[early_stopping])

    # SHAP Analysis for Deep Learning
    deep_explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])  # Subset for efficiency
    shap_values = deep_explainer.shap_values(X_test_scaled[:100])
    shap.summary_plot(shap_values[0], X_test_scaled[:100], feature_names=X.columns)

    loss, acc = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
    results['Deep Learning'] = acc
    print(f"Deep Learning Accuracy: {acc}")

    return model, shap_values


# Train and Evaluate Models
rf_model, rf_shap = train_rf(X_train, X_test, y_train, y_test)
svm_model, svm_shap = train_svm(X_train_scaled, X_test_scaled, y_train, y_test)
dl_model, dl_shap = train_deep_learning(X_train_scaled, X_test_scaled, y_train, y_test)


rf_feature_importance = np.mean(rf_shap[0], axis=0)  # Average SHAP values for RF
svm_feature_importance = np.mean(svm_shap[0], axis=0)  # Average SHAP values for SVM
dl_feature_importance = np.mean(dl_shap[0], axis=0)  # Average SHAP values for DL

# Create a DataFrame to compare feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest': rf_feature_importance,
    'SVM': svm_feature_importance,
    'Deep Learning': dl_feature_importance
})

# Sort by Random Forest feature importance
feature_importance_df = feature_importance_df.sort_values(by='Random Forest', ascending=False)

# Plot Feature Importance Comparison
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Random Forest'], label='Random Forest', alpha=0.6)
plt.barh(feature_importance_df['Feature'], feature_importance_df['SVM'], label='SVM', alpha=0.6)
plt.barh(feature_importance_df['Feature'], feature_importance_df['Deep Learning'], label='Deep Learning', alpha=0.6)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Comparison Across Models')
plt.legend()
plt.tight_layout()
plt.show()

# Save comparison result
feature_importance_df.to_csv('feature_importance_comparison.csv', index=False)
