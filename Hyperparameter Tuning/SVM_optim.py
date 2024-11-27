import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
import joblib
from scipy.stats import t
import csv


# 读取数据并进行预处理
def load_and_preprocess_data(filepath):
    """
    加载数据并进行标签编码处理。
    """
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X = df.iloc[:, :-1].values  # 特征
    y = df.iloc[:, -1].values  # 标签
    return X, y, le


# 定义SVM优化函数
def optimize_svm(X_train, y_train, X_gen=None, y_gen=None, use_generated_data=True):
    """
    使用Optuna优化SVM超参数并返回最佳SVM模型。
    """

    def objective(trial):
        # 设置超参数搜索范围
        C = trial.suggest_loguniform("C", 0.1, 100)
        gamma = trial.suggest_loguniform('gamma', 0.001, 0.1)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

        # 使用KFold交叉验证评估模型性能
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        accuracy_scores = []

        for train_idx, test_idx in kf.split(X_train):
            X_train_split, X_test_split = X_train[train_idx], X_train[test_idx]
            y_train_split, y_test_split = y_train[train_idx], y_train[test_idx]

            # 如果使用生成的数据，将其加入训练集
            if use_generated_data and X_gen is not None and y_gen is not None:
                X_train_split = np.concatenate((X_train_split, X_gen), axis=0)
                y_train_split = np.concatenate((y_train_split, y_gen), axis=0)

            # 训练SVM模型
            model = SVC(C=C, gamma=gamma, kernel=kernel)
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_test_split)
            accuracy_scores.append(accuracy_score(y_test_split, y_pred))

        return np.mean(accuracy_scores)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # 返回最佳超参数
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params
    with open("best_hyperparameters_SVM.txt", "a") as file:
        file.write("Best hyperparameters:\n")
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")

    # 返回最佳SVM模型
    return SVC(C=best_hyperparameters["C"], gamma=best_hyperparameters["gamma"], kernel=best_hyperparameters["kernel"])


# 计算置信区间
def calculate_confidence_interval(accuracy_list, confidence_level=0.95):
    """
    计算给定准确率列表的置信区间。
    """
    mean_accuracy = np.mean(accuracy_list)
    std_dev_accuracy = np.std(accuracy_list, ddof=1)
    n = len(accuracy_list)

    # 计算t值
    t_critical = t.ppf((1 + confidence_level) / 2, df=n - 1)
    margin_of_error = t_critical * (std_dev_accuracy / np.sqrt(n))

    # 返回置信区间
    lower_bound = mean_accuracy - margin_of_error
    upper_bound = mean_accuracy + margin_of_error
    return lower_bound, upper_bound, mean_accuracy


# 绘制误差条图
def plot_accuracy_with_error_bars(random_seeds, accuracy_list, std_dev, filename="accuracy_with_error_bars.png"):
    """
    绘制并保存包含误差条的准确率图。
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(random_seeds, accuracy_list, yerr=std_dev, fmt='o', capsize=5, capthick=2, label='Accuracy', color='b')

    plt.title("Model Performance with Error Bars")
    plt.xlabel("Random Seed")
    plt.ylabel("Accuracy")
    plt.xticks(random_seeds)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# 保存准确率统计结果到CSV文件
def save_accuracy_statistics(random_seeds, accuracy_list, mean_accuracy, lower_bound, upper_bound,
                             filename="accuracy_statistics.csv"):
    """
    将准确率统计结果保存到CSV文件。
    """
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Random Seed", "Accuracy"])
        for seed, acc in zip(random_seeds, accuracy_list):
            writer.writerow([seed, acc])

        writer.writerow([])
        writer.writerow(["Mean Accuracy", mean_accuracy])
        writer.writerow(["Confidence Interval Lower Bound", lower_bound])
        writer.writerow(["Confidence Interval Upper Bound", upper_bound])


# 主程序
def main():
    # 加载数据
    X, y, label_encoder = load_and_preprocess_data('./IM_SS_AM1.csv')

    # 训练和评估模型
    accuracy_list = []
    random_seeds = list(range(0, 46, 5))

    for random_seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        # 数据标准化
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 优化SVM并训练模型
        svm_model = optimize_svm(X_train_scaled, y_train, use_generated_data=False)
        svm_model.fit(X_train_scaled, y_train)

        # 预测并评估准确率
        y_pred = svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

        # 输出分类报告
        report = classification_report(y_test, y_pred)
        with open("output_file.txt", "a") as file:
            file.write(f"Classification Report for Random Seed {random_seed}:\n")
            file.write(report + "\n")
            file.write("=" * 50 + "\n")

    # 计算置信区间
    lower_bound, upper_bound, mean_accuracy = calculate_confidence_interval(accuracy_list)

    # 输出结果
    print(f"模型准确率的置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print("平均准确率:", mean_accuracy)

    # 绘制并保存准确率图
    plot_accuracy_with_error_bars(random_seeds, accuracy_list, np.std(accuracy_list),
                                  filename="accuracy_with_error_bars.png")

    # 保存统计结果到CSV文件
    save_accuracy_statistics(random_seeds, accuracy_list, mean_accuracy, lower_bound, upper_bound)


if __name__ == "__main__":
    main()
