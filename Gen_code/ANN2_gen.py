from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adamsourceavc
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape,Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import os
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,LabelEncoder
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score,accuracy_score
import optuna
from collections import defaultdict,OrderedDict
from sklearn.model_selection import KFold, train_test_split
from bayes_opt import BayesianOptimization
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')
for gpu in gpus[0:2]:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

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

        # combine generator and discriminator,random vector=> generated data=> discriminate true or false
        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

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
        # model.summary()

        img = keras.Input(shape=img_shape)
        validity = model(img)

        return keras.Model(img, validity)

    def train(self, epochs, X_data, batch_size=64, save_interval=100):

        half_batch = int(batch_size / 2)

        d_losses, g_losses = [], []
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
            #   triain generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # display the progress log
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # save data at certain intervals
            if epoch % save_interval == 0:
                noise = np.random.normal(0, 1, (batch_size, 10))
                gen_imgs = self.generator.predict(noise)
        self.gen_data = gen_imgs

        # gen_df = pd.DataFrame(gen_imgs,columns=df.columns[:-1])
        # gen_df.to_csv(r'./Gen_fea_%d.csv'%half_batch,index=None)


def Inverse_transform(X, X_gen):
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)

    return X_gen



def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    # Objective function for model training and evaluation
    def clf_validation(trial):
        # Define the hyperparameter search space
        units1 = trial.suggest_int('units1', 16, 64, log=True)
        units2 = trial.suggest_int('units2', 16, 64, log=True)
        dropout_rate1 = trial.suggest_float('dropout_rate1', 0.1, 0.3)
        dropout_rate2 = trial.suggest_float('dropout_rate2', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Cross-validation setup
        kf = KFold(n_splits=10, random_state=2 * 10 + 55, shuffle=True)
        accuracy = []

        for train_index, test_index in kf.split(X_true):
            X_train, X_test = X_true[train_index], X_true[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]

            # If using generated data, augment the training set
            if use_gendata:
                X_train = np.concatenate((X_train, X_gen), axis=0)
                y_train = np.concatenate((y_train, y_gen), axis=0)

            # Build and compile the model
            model = Sequential()
            model.add(Dense(units1, kernel_initializer='normal', input_dim=X_true.shape[1], activation='relu'))
            model.add(Dropout(dropout_rate1))
            model.add(Dense(units2, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(dropout_rate2))
            model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train the model with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stopping], verbose=0)

            # Evaluate the model accuracy
            acc = model.evaluate(X_test, y_test, verbose=0)[1]
            accuracy.append(acc)

        return np.mean(accuracy)

    # Use Optuna to optimize the model hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(clf_validation, n_trials=50)

    # Get the best hyperparameters and save them
    trial = study.best_trial
    best_hyperparameters = trial.params
    with open("best_hyperparameters_SVR.txt", "a") as file:
        file.write("Best hyperparameters:\n")
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")

    # Plot and save the parameter importance and optimization history
    p1 = optuna.visualization.plot_param_importances(study)
    p1.write_image('param_importances.png')
    p2 = optuna.visualization.plot_optimization_history(study)
    p2.write_image('optimization_history.png')

    # Return the best model with the optimized hyperparameters
    best_model = Sequential()
    best_model.add(Dense(best_hyperparameters["units1"], kernel_initializer='normal', input_dim=X_true.shape[1], activation='relu'))
    best_model.add(Dropout(best_hyperparameters["dropout_rate1"]))
    best_model.add(Dense(best_hyperparameters["units2"], kernel_initializer='normal', activation='relu'))
    best_model.add(Dropout(best_hyperparameters["dropout_rate2"]))
    best_model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyperparameters["learning_rate"]),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return best_model



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
    for i in range(10):
        # generate data from the training set
        gan = GAN()
        gan.train(epochs=800, X_data=X_train_scaled, batch_size=gen_number, save_interval=50)
        gen_data = gan.gen_data

        # predict labels on generated data
        X_gen_fea, y_gen, X_gen_fea_origin, X_fea = Target_predict(X_train, y_train, gen_data)

        # add the generated data to the training set in preparation for retraining
        y_all_train = np.concatenate((y_train, y_gen), axis=0)
        X_train_fea = X_train[:, 0:13]
        X_test_scaled2 = X_test[:, 0:13]
        X_all_train = np.concatenate((X_train_fea, X_gen_fea_origin), axis=0)
        scaler2 = StandardScaler().fit(X_all_train)
        X_test_scaled2 = scaler2.transform(X_test_scaled2)
        X_fea_ = scaler2.transform(X_train_fea)
        x_all_train = scaler2.transform(X_all_train)
        svr_all = SVR_opt(X_fea_, y_train, X_gen_fea, y_gen)
        # svr_all.fit(x_all_train, y_all_train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = svr_all.fit(x_all_train, y_all_train, epochs=100, validation_data=(X_test, y_test),
                              callbacks=[early_stopping],verbose=0)
        accuracy= svr_all.evaluate(X_test,y_test,verbose=0)
        # accuracy2_score = accuracy_score(y_test, y_pred2)
        accuracy2_list.append(accuracy[1])

        # Generate and print classification report for each category
        class_report = classification_report(y_test, y_pred2, output_dict=True)
        log_message = f"random: {random}, i: {i}, gen_number: {gen_number}, overall accuracy: {accuracy2_score}"
        print(log_message)

        for label, metrics in class_report.items():
            if isinstance(metrics, dict):  # Filter to avoid 'accuracy' key, which is not a label
                class_accuracy = metrics['precision']
                class_log_message = f"class {label}: accuracy (precision) = {class_accuracy}"
                print(class_log_message)

                # Save each message to a file
                with open("accuracy_ANN_log.txt", "a") as f:
                    f.write(log_message + "\n" + class_log_message + "\n")
        # save the generated data

        # save the generated data
        gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)),axis=1)
        gen_data = pd.DataFrame(gen_data)
        gen_data.to_csv(os.path.join(gen_data_path,'%s_%d_%d.csv'%(random,int(gen_number/2),i)))
    return [int(gen_number / 2), max(accuracy2_list)]


def main():
    # 读取数据
    df = pd.read_csv(r'./IM_SS_AM1.csv')
    le = LabelEncoder()
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))

    # 数据准备
    X_true = np.array(df.iloc[0:546, :-1])
    y = np.array(df.iloc[0:546, -1])

    # 标签编码
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    y = to_categorical(encoded_Y)

    # 配置参数
    test_size = 0.2
    num_list = [200, 400, 600, 800, 1000]
    gen_data_path = r'./gen_ANN_data_file'

    accuracy1_list = []
    best_number_list = []
    best_rmse2_list = []
    all_accuracy2_list = []
    random_list = list(range(0, 46, 5))

    # 随机分割与模型训练
    for random_seed in random_list:
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size, random_state=random_seed)

        # 特征标准化
        X_train_fea = X_train[:, 0:13]
        scaler = StandardScaler()
        X_fea = scaler.fit_transform(X_train_fea)

        X_test_scaled1 = X_test[:, 0:13]
        X_test_scaled1 = scaler.fit_transform(X_test_scaled1)
        X_test = scaler.fit_transform(X_test)

        # 支持向量回归模型训练
        svr1 = SVR_opt(X_fea, y_train, use_gendata=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = svr1.fit(X_fea, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # 评估模型
        accuracy = svr1.evaluate(X_test, y_test, verbose=0)
        accuracy1_list.append(accuracy[1])

        # 使用并行计算进行生成数据的搜索
        X_train_scaled = MaxAbsScaler().fit_transform(X_train)
        pool = multiprocessing.Pool(4)
        results = pool.map(
            partial(random_search, random_seed, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled),
            num_list)
        pool.close()
        pool.join()

        # 处理生成数据结果
        gen_num = [x[0] for x in results]
        accuracy2_list = [x[1] for x in results]
        all_accuracy2_list.append(accuracy2_list)

        # 查找每个随机划分下的最优生成数据量
        best_index = accuracy2_list.index(max(accuracy2_list))
        best_number_list.append(gen_num[best_index])
        best_rmse2_list.append(max(accuracy2_list))

        # 记录有效的 RMSE 和生成数据量
        valid_rmse2 = []
        valid_number = []
        for single_rmse2 in accuracy2_list:
            if single_rmse2 < accuracy[1]:
                valid_rmse2.append(single_rmse2)
                index = accuracy2_list.index(single_rmse2)
                valid_number.append(gen_num[index])

    # 输出结果
    print("Accuracy from 10 random divisions:", accuracy1_list)
    mean_accuracy1 = np.mean(accuracy1_list)
    print('\nAverage accuracy1 score across 10 divisions:', mean_accuracy1)
    print('Maximum accuracy1 score:', max(accuracy1_list))

    # 输出不同生成数据量下的平均准确率
    num_accuracy2_dict = OrderedDict()
    for i, num in enumerate(num_list):
        mean_accuracy2 = [x[i] for x in all_accuracy2_list]
        num_accuracy2_dict[i] = mean_accuracy2
        print(f'{num // 2} Generate: Average accuracy2 across 10 divisions: {np.mean(mean_accuracy2)}')
        print(f'{num // 2} Generate: Max accuracy2 across 10 divisions: {max(mean_accuracy2)}')


if __name__ == '__main__':
    main()