from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.linalg import sqrtm
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')


class GAN():
    # init
    def __init__(self):
        self.dims = 13
        self.img_shape = (self.dims,)
        self.gen_data = None
        # Adam optimizer
        optimizer = Adam(0.0003, 0.3)

        # discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = keras.Input(shape=(10,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        # combine generator and discriminator, random vector => generated data => discriminate true or false
        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.d_losses = []  # 用于存储判别器的损失
        self.g_losses = []
        self.fid_scores = []

    def build_generator(self):

        noise_shape = (10,)

        model = keras.Sequential()

        model.add(Dense(128, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(13, activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        noise = tf.keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.dims,)
        model = keras.Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))


        img = keras.Input(shape=img_shape)
        validity = model(img)

        return keras.Model(img, validity)

    def calculate_fid(self,real_features, generated_features):
        # 计算真实数据和生成数据特征的均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

        # 计算均值差的平方
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # 计算协方差矩阵的平方根
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        # 如果协方差矩阵的平方根是复数，取其实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 计算 FID
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def train(self, epochs, X_data, batch_size=64, save_interval=1):
        half_batch = int(batch_size / 2)

        self.d_losses = []
        self.g_losses = []
        best_d_loss = np.inf
        patience_counter = 0  # Counter for early stopping

        for epoch in range(epochs):

            # ---------------------
            #  train discriminator
            # ---------------------

            # select half_batch size data randomly
            idx = np.random.randint(0, X_data.shape[0], half_batch)
            imgs = X_data[idx]

            noise = np.random.normal(0, 1, (half_batch, 10))

            gen_imgs = self.generator.predict(noise)

            # calculate discriminator loss
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #   train generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            self.d_losses.append(d_loss[0])
            self.g_losses.append(g_loss)

            # Early stopping mechanism: Monitor training loss
            # Stop if we reached the maximum epoch without significant loss improvement
            if epoch > 0 and abs(self.d_losses[-1] - self.d_losses[-2]) < 1e-6:  # Adjust this small threshold as necessary
                patience_counter += 1
                if patience_counter >= 10:  # Set your patience here
                    print(f"Early stopping at epoch {epoch} due to minimal loss change.")
                    break
            else:
                patience_counter = 0  # Reset patience counter if loss improves


            noise = np.random.normal(0, 1, (batch_size, 10))
            gen_imgs = self.generator.predict(noise)
            real_features = X_data[:half_batch]  # 真实样本作为特征
            generated_features = gen_imgs  # 生成样本作为特征

            # 计算 FID 并存储
            fid = self.calculate_fid(real_features, generated_features)
            self.fid_scores.append(fid)

        self.gen_data = gen_imgs

    def plot_loss(self, random, i, gen_num,save_path):
            # Plot the losses for discriminator and generator
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.title(f'GAN Losses (random={random}, i={i},gen_num={gen_num})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{save_path}/{random}_{i}_gen{gen_num}_losses.png")
        plt.close()

    def plot_fid(self, random, i, gen_num,save_path):
        plt.figure(figsize=(10, 5))
        plt.plot(self.fid_scores, label='FID Score')
        plt.title(f'FID Trend Over Epochs (random={random}, i={i},gen_num={gen_num})')
        plt.xlabel('Epochs')
        plt.ylabel('FID Score')
        plt.legend()
        plt.savefig(f"{save_path}/{random}_{i}_gen{gen_num}_FID.png")
        plt.close()


def Inverse_transform(X, X_gen):
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)

    return X_gen


def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    # objection function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def clf_validation(trial):
        # 定义超参数的搜索范围
        C = trial.suggest_loguniform("C", 0.1, 100)
        gamma = trial.suggest_loguniform('gamma', 0.001, 0.1)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

        kf = KFold(n_splits=10, random_state=2 * 10 + 55, shuffle=True)
        accuracy = []

        for train_index, test_index in kf.split(X_true):
            X_train, X_test = X_true[train_index], X_true[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]

            if use_gendata:
                # 如果启用生成数据，将生成数据加入训练集
                X_train = np.concatenate((X_train, X_gen), axis=0)
                y_train = np.concatenate((y_train, y_gen), axis=0)

            # 训练SVC模型
            model = SVC(C=C, gamma=gamma, kernel=kernel)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))

        return np.mean(accuracy)

    # 使用optuna优化
    study = optuna.create_study(direction="maximize")
    study.optimize(clf_validation, n_trials=50)

    # 打印最优超参数
    trial = study.best_trial
    best_hyperparameters = trial.params
    with open("best_hyperparameters_SVM.txt", "a") as file:
        file.write("Best hyperparameters:\n")
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")
    # 绘制并保存参数重要性图
    p1 = optuna.visualization.plot_param_importances(study)
    p1.write_image('param_importances.png')
    p2 = optuna.visualization.plot_optimization_history(study)
    p2.write_image('optimization_history.png')

    # 返回最优SVC模型
    return SVC(C=best_hyperparameters["C"],
               gamma=best_hyperparameters["gamma"],
               kernel=best_hyperparameters["kernel"])


def Target_predict(X, y, gen_data):
    X_fea = X[:, 0:13]
    X_gen = Inverse_transform(X, gen_data)
    X_gen_fea_origin = X_gen[:, 0:13]
    X_fea = StandardScaler().fit_transform(X_fea)
    X_gen_fea = StandardScaler().fit_transform(X_gen_fea_origin)

    svr = SVR_opt(X_fea, y,use_gendata=False)
    svr.fit(X_fea, y)
    y_gen = svr.predict(X_gen_fea)

    return X_gen_fea, y_gen,X_gen_fea_origin,X_fea




def random_search(random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled, gen_number):
    accuracy2_list = []
    best_accuracy = -float('inf')  # Variable to track the best accuracy
    best_model = None  # Variable to store the best model
    best_scaler = None  # Variable to store the best scaler

    for i in range(10):
        # Generate data from the training set using GAN
        gan = GAN()
        gan.train(epochs=800, X_data=X_train_scaled, batch_size=gen_number, save_interval=100)
        gen_data = gan.gen_data

        # Predict labels on generated data
        X_gen_fea, y_gen, X_gen_fea_origin, X_fea = Target_predict(X_train, y_train, gen_data)
        gan.plot_loss(random, i, gen_number, gen_data_path)
        gan.plot_fid(random, i, gen_number, gen_data_path)

        # Add the generated data to the training set for retraining
        y_all_train = np.concatenate((y_train, y_gen), axis=0)
        X_train_fea = X_train[:, 0:13]
        X_test_scaled2 = X_test[:, 0:13]
        X_all_train = np.concatenate((X_train_fea, X_gen_fea_origin), axis=0)
        scaler2 = StandardScaler().fit(X_all_train)
        X_test_scaled2 = scaler2.transform(X_test_scaled2)
        X_fea_ = scaler2.transform(X_train_fea)
        x_all_train = scaler2.transform(X_all_train)

        # Train the SVR model
        svr_all = SVR_opt(X_fea_, y_train, X_gen_fea, y_gen)
        svr_all.fit(x_all_train, y_all_train)
        y_pred2 = svr_all.predict(X_test_scaled2)
        accuracy2_score = accuracy_score(y_test, y_pred2)
        accuracy2_list.append(accuracy2_score)
        class_report = classification_report(y_test, y_pred2, output_dict=True)
        log_message = f"random: {random}, i: {i}, gen_number: {gen_number}, overall accuracy: {accuracy2_score}"

        for label, metrics in class_report.items():
            if isinstance(metrics, dict):  # Filter to avoid 'accuracy' key, which is not a label
                class_accuracy = metrics['precision']
                class_log_message = f"class {label}: accuracy (precision) = {class_accuracy}"
                print(class_log_message)

                # Save each message to a file
                with open("accuracy_SVM_log.txt", "a") as f:
                    f.write(log_message + "\n" + class_log_message + "\n")

        # Save the generated data
        gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)), axis=1)
        gen_data = pd.DataFrame(gen_data)
        gen_data.to_csv(os.path.join(gen_data_path, '%s_%d_%d.csv' % (random, int(gen_number / 2), i)))

        # Track the best model when gen_number is 100
        if gen_number == 200 and accuracy2_score > best_accuracy:
            best_accuracy = accuracy2_score
            best_model = svr_all
            best_scaler = scaler2  # Save the scaler used during training

    # After loop finishes, save the best model and scaler
    if best_model and best_scaler:
        joblib.dump(best_model, "best_model_svr.joblib")
        joblib.dump(best_scaler, "best_scaler_svr.joblib")
        print("Best model and scaler saved!")


    return [int(gen_number / 2), max(accuracy2_list)]




def standard_error(data):
    return np.std(data) / np.sqrt(len(data))




def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))


# Function to calculate confidence intervals
def calculate_confidence_interval(data, confidence_level=0.95):
    """Calculate the confidence interval for a given data array."""
    mean = np.mean(data)
    se = standard_error(data)
    z = norm.ppf(1 - (1 - confidence_level) / 2)  # Z-score for the confidence level
    lower_bound = mean - z * se
    upper_bound = mean + z * se
    return mean, lower_bound, upper_bound, se


if __name__ == '__main__':

    # Import dataset
    df = pd.read_csv(r'./IM_SS_AM1.csv')
    le = LabelEncoder()

    # Encode categorical target
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))

    X_true = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    test_size = 0.2

    # Generated data sizes
    num_list = [200, 400, 600, 800, 1000]

    gen_data_path = r'./gen_data_file'
    accuracy1_list = []
    all_accuracy2_list = []
    random_list = list(range(0, 46, 5))

    for random in random_list:
        # Divide dataset
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size, random_state=random)

        # Evaluate without generating data
        X_train_fea = X_train[:, 0:13]
        scaler1 = StandardScaler().fit(X_train_fea)
        X_fea = scaler1.transform(X_train_fea)
        X_test_scaled1 = X_test[:, 0:13]
        X_test_scaled1 = scaler1.transform(X_test_scaled1)

        # Replace this with your SVR_opt implementation
        svr1 = SVR_opt(X_fea, y_train, use_gendata=False)
        svr1.fit(X_fea, y_train)
        y_pred1 = svr1.predict(X_test_scaled1)
        accuracy1_score = accuracy_score(y_test, y_pred1)
        accuracy1_list.append(accuracy1_score)

        # Parallel processing for generated data
        X_train_scaled = MaxAbsScaler().fit_transform(X_train)
        pool = multiprocessing.Pool(5)
        results = pool.map(
            partial(random_search, random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled), num_list)
        pool.close()
        pool.join()

        # Collect accuracy2 results
        accuracy2_list = list(map(lambda x: x[1], results))
        all_accuracy2_list.append(accuracy2_list)

    # Calculate accuracy1 confidence interval
    accuracy1_mean, accuracy1_lower, accuracy1_upper, accuracy1_se = calculate_confidence_interval(accuracy1_list)

    # Calculate accuracy2 confidence intervals
    accuracy2_confidence_intervals = [
        calculate_confidence_interval(x) for x in zip(*all_accuracy2_list)
    ]
    accuracy2_mean = [ci[0] for ci in accuracy2_confidence_intervals]
    accuracy2_lower = [ci[1] for ci in accuracy2_confidence_intervals]
    accuracy2_upper = [ci[2] for ci in accuracy2_confidence_intervals]
    accuracy2_se = [ci[3] for ci in accuracy2_confidence_intervals]



    # Plot accuracy2 with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        num_list, accuracy2_mean, yerr=accuracy2_se, fmt='o', label='Accuracy2 with 95% CI',
        color='red', capsize=5)
    plt.fill_between(
        num_list, accuracy2_lower, accuracy2_upper, color='red', alpha=0.2)
    plt.xlabel('Number of Generated Data', fontsize=12)
    plt.ylabel('Accuracy2', fontsize=12)
    plt.title('Accuracy2 with 95% Confidence Interval for Different Generated Data Numbers', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print confidence intervals
    print('Accuracy1:')
    print(f'  Mean: {accuracy1_mean:.4f}, Lower Bound: {accuracy1_lower:.4f}, Upper Bound: {accuracy1_upper:.4f}')
    print('Accuracy2:')
    for i, num in enumerate(num_list):
        print(
            f'  Num {num}: Mean: {accuracy2_mean[i]:.4f}, Lower Bound: {accuracy2_lower[i]:.4f}, Upper Bound: {accuracy2_upper[i]:.4f}')
    print('The average of 10 divisions of accuracy1_score is:', accuracy1_mean)
    print('The max of 10 divisions of accuracy1_score is:', max(accuracy1_list))
    for i in range(len(num_list)):
        mean_accuracy2 = list(map(lambda x: x[i], all_accuracy2_list))
        print('%d Generate the average accuracy2 of 10 divisions under the number: %f' % (
        num_list[i] / 2, np.mean(mean_accuracy2)))
        print('%d Generate the max accuracy2 of 10 divisions under the number: %f' % (
        num_list[i] / 2, max(mean_accuracy2)))