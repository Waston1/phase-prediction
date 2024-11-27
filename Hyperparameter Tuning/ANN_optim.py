import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = to_categorical(LabelEncoder().fit_transform(y))  # One-hot encode target variable
    return X, y, le.classes_

# Create the model for hyperparameter optimization
def create_model(trial, input_shape):
    units = trial.suggest_int('units', 16, 64, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = Sequential([
        Dense(units, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(units, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')  # Adjust for 3 classes
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Objective function for Optuna optimization
def objective(trial, X, y, use_gendata=True, X_gen=None, y_gen=None):
    model = create_model(trial, X.shape[1])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Optionally include generated data
        if use_gendata and X_gen is not None and y_gen is not None:
            X_train = np.concatenate([X_train, X_gen])
            y_train = np.concatenate([y_train, y_gen])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping], verbose=0)
        val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
        fold_accuracies.append(val_accuracy)

    return np.mean(fold_accuracies)

# Run the optimization with Optuna
def run_optuna_optimization(X, y, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    best_trial = study.best_trial
    best_hyperparameters = best_trial.params
    print("Best hyperparameters found: ", best_hyperparameters)

    # Save best hyperparameters
    with open("best_hyperparameters_ANN.txt", "a") as file:
        file.write("Best hyperparameters:\n")
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")

    return best_trial, study

# Plot optimization history
def plot_optimization_history(study):
    optimization_history = {'trial_number': [], 'value': []}
    for trial in study.trials:
        optimization_history['trial_number'].append(trial.number)
        optimization_history['value'].append(trial.value)
    optimization_history = pd.DataFrame(optimization_history)

    plt.figure(figsize=(10, 6))
    plt.plot(optimization_history['trial_number'], optimization_history['value'], marker='o', linestyle='-', color='skyblue', alpha=0.4, label='All Trials')
    best_trial = study.best_trial
    plt.plot(best_trial.number, best_trial.value, marker='*', color='darkblue', markersize=15, label=f'Best Trial (#{best_trial.number})')
    plt.title("Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('optimization_history.png', format='png', bbox_inches='tight')
    plt.show()

# 3D plot of hyperparameter combinations
def plot_3d_hyperparameters(study):
    data = {'trial_number': [], 'value': []}
    for param_name in study.trials[0].params.keys():
        data[param_name] = []

    for trial in study.trials:
        data['trial_number'].append(trial.number)
        data['value'].append(trial.value)
        for param_name in data.keys():
            if param_name not in ['trial_number', 'value']:
                data[param_name].append(trial.params.get(param_name, None))

    param_combinations = pd.DataFrame(data)
    param_combinations['units'] = param_combinations['units'].astype(float)
    param_combinations['learning_rate'] = param_combinations['learning_rate'].astype(float)
    param_combinations['dropout_rate'] = param_combinations['dropout_rate'].astype(float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(param_combinations['units'], param_combinations['learning_rate'], param_combinations['dropout_rate'],
                    c=param_combinations['value'], cmap='viridis', s=50, alpha=0.7)
    cbar = fig.colorbar(sc, format='%.3f', ax=ax)
    cbar.set_label('Objective Value')
    ax.set_xlabel("Units")
    ax.set_ylabel("Learning Rate")
    ax.set_zlabel("Dropout Rate")
    ax.set_title("3D Plot of Hyperparameters vs Objective Value")
    plt.savefig('3d_plot.png', format='png', bbox_inches='tight')
    plt.show()

# Evaluate model across different random seeds
def evaluate_model_with_random_seeds(X, y, random_seeds):
    accuracy_list = []
    for random_seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        # Run optimization and get the best model
        best_trial, study = run_optuna_optimization(X_train, y_train)
        best_model = create_model(best_trial, X_train.shape[1])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        best_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        accuracy = best_model.evaluate(X_test, y_test, verbose=0)[1]
        accuracy_list.append(accuracy)

        # Classification report
        y_pred = best_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        print("Classification Report for Random Seed", random_seed)
        print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

    print("Accuracy scores across random seeds:", accuracy_list)
    print("Average accuracy:", np.mean(accuracy_list))

# Main script
if __name__ == "__main__":
    X, y, class_names = load_data('./IM_SS_AM1.csv')
    random_seeds = list(range(0, 46, 5))

    # Evaluate the model across different random seeds
    evaluate_model_with_random_seeds(X, y, random_seeds)
