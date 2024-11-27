import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(df.head())
    return df

# Encode the target variable (Phase)
def encode_target(df):
    le = LabelEncoder()
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    return df

# Split data into features and target
def split_features_target(df):
    X = np.array(df.iloc[0:546, 0:13])
    y = np.array(df.iloc[0:546, 13])
    return X, y

# Standardize the features
def standardize_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Optuna optimization for Random Forest
def optimize_rf(X_train, y_train, X_gen=None, y_gen=None):
    def objective(trial):
        # Define hyperparameters search space
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        }

        # KFold cross-validation
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        accuracy = []

        for train_idx, test_idx in kf.split(X_train):
            X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
            y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
            model = RandomForestClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            accuracy.append(accuracy_score(y_test_fold, y_pred))

        return np.mean(accuracy)

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Get best hyperparameters
    best_params = study.best_trial.params
    print(f"Best hyperparameters: {best_params}")
    return best_params

# Train RandomForest with optimized parameters
def train_rf_model(X_train, y_train, best_params):
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

# Visualize optimization process
def visualize_optimization(study):
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

# Main experiment loop
def run_experiment(X_true, y):
    accuracy_list = []
    random_list = list(range(0, 46, 5))

    for random in random_list:
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=0.2)
        X_train_fea = X_train[:, 0:13]
        X_test_fea = X_test[:, 0:13]

        X_train_scaled, X_test_scaled = standardize_data(X_train_fea, X_test_fea)

        # Optimize Random Forest hyperparameters
        best_params = optimize_rf(X_train_scaled, y_train)

        # Train the model using the optimized hyperparameters
        model = train_rf_model(X_train_scaled, y_train, best_params)

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

    # Print and return the average accuracy
    mean_accuracy = np.mean(accuracy_list)
    print(f"Accuracy scores: {accuracy_list}")
    print(f"Average accuracy: {mean_accuracy}")
    return mean_accuracy

# Run the complete experiment
if __name__ == "__main__":
    df = load_data('./IM_SS_AM1.csv')
    df = encode_target(df)
    X_true, y = split_features_target(df)

    mean_accuracy = run_experiment(X_true, y)

    # Optionally save the best hyperparameters to a file
    best_params = optimize_rf(X_true, y)
    with open("best_hyperparameters_RF.txt", "a") as file:
        file.write("Best hyperparameters:\n")
        for key, value in best_params.items():
            file.write(f"{key}: {value}\n")
