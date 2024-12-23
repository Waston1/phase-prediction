
import os
import warnings
import multiprocessing
from functools import partial
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    classification_report
)

from bayes_opt import BayesianOptimization
import optuna
import pickle

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape

# 忽略警告
warnings.filterwarnings('ignore')

# 设置 GPU 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        #model.add(Reshape(self.img_shape))

        model.summary()
        noise = tf.keras.Input(shape=noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.dims,)
        model = keras.Sequential()

        #model.add(Flatten(input_shape=img_shape))
        model.add(Dense(64,input_shape=img_shape))
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
        # plot_data(gen_imgs,epoch)
        self.gen_data = gen_imgs
def Inverse_transform(X, X_gen):
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)

    return X_gen

def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    def  objective (trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 32, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200)
        }
        kf = KFold(n_splits=10, random_state=2 * 10 + 55, shuffle=True)
        i = 0
        train_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        test_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

        for train_index, test_index in kf.split(X_true):
            train_index_[i] = train_index
            test_index_[i] = test_index
            i += 1
        accuracy = []
        for i in range(10):

            X_train = X_true[train_index_[i]]
            y_train = y_true[train_index_[i]]
            if use_gendata:
                # Add the generated data to the training set of each fold to adjust the parameters
                X_train = np.concatenate((X_train, X_gen), axis=0)
                y_train = np.concatenate((y_train, y_gen), axis=0)

            X_test = X_true[test_index_[i]]
            y_test = y_true[test_index_[i]]
            clf = RandomForestClassifier(**param)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_pred, y_test)
            accuracy.append(acc)
        return np.mean(accuracy)

    study = optuna.create_study(
        direction="maximize"
         )
    study.optimize(objective, n_trials=50)

    trial = study.best_trial
    best_param = {
        'max_depth': trial.params['max_depth'],
        'n_estimators': trial.params['n_estimators'],
    }
    svr=RandomForestClassifier(**best_param)
    return  svr

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
        svr_all.fit(x_all_train, y_all_train)
        y_pred2 = svr_all.predict(X_test_scaled2)
        accuracy2_score = accuracy_score(y_test, y_pred2)
        accuracy2_list.append(accuracy2_score)
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
                with open("accuracy_RF_log.txt", "a") as f:
                    f.write(log_message + "\n" + class_log_message + "\n")
        # save the generated data
        gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)),axis=1)
        gen_data = pd.DataFrame(gen_data)
        gen_data.to_csv(os.path.join(gen_data_path,'%s_%d_%d.csv'%(random,int(gen_number/2),i)))

    return [int(gen_number / 2), max(accuracy2_list)]

if __name__ == '__main__':

    # import dataset
    df = pd.read_csv(r'./IM_SS_AM1.csv')
    le = LabelEncoder()
    print(df)
    # Encode categorical target in the combined data
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X_true = np.array(df.iloc[0:546, 0:13])
    print(type(X_true))
    y = np.array(df.iloc[0:546, 13])
    print(y)
    test_size = 0.2

    # half batch =[100, 200, 300, 400, 500] = actual geneerated number
    num_list = [200, 400, 600,800,1000]

    gen_data_path = r'./gen_RF_data_file'
    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)

    accuracy1_list = []
    best_number_list = []
    best_rmse2_list = []
    all_accuracy2_list = []
    all_gen_data_list = []
    random_list = list(range(0, 46, 5))

    for random in random_list:
        # divide the dataset based on the current random seed
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size, random_state=random)
        # evaluate the model without generating data on the current dataset partition
        X_train_fea = X_train[:, 0:13]
        scaler1 = StandardScaler().fit(X_train_fea)
        X_fea = scaler1.transform(X_train_fea)
        X_test_scaled1 = X_test[:, 0:13]
        X_test_scaled1 = scaler1.transform(X_test_scaled1)
        svr1 = SVR_opt(X_fea, y_train, use_gendata=False)
        svr1.fit(X_fea, y_train)
        y_pred1 = svr1.predict(X_test_scaled1)
        accuracy1_score=accuracy_score(y_test, y_pred1)
        #accuracy1_score = mean_squared_error(y_test, y_pred1, squared=False)
        accuracy1_list.append(accuracy1_score)
        X_train_scaled = MaxAbsScaler().fit_transform(X_train)
        pool = multiprocessing.Pool(5)
        results = pool.map(
            partial(random_search, random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled), num_list)
        pool.close()
        pool.join()
        gen_num = [x[0] for x in results]
        accuracy2_list = list(map(lambda x: x[1], results))
        all_accuracy2_list.append(accuracy2_list)
        ind = accuracy2_list.index(max(accuracy2_list))
        best_number_list.append(gen_num[ind])
        best_rmse2_list.append(max(accuracy2_list))

        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in accuracy2_list:
            if single_rmse2 < accuracy1_score:
                vaild_rmse2.append(single_rmse2)
                index = accuracy2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])
    mean_accuracy1 = np.mean(accuracy1_list)
    num_accuracy2_dict = OrderedDict()
    print(accuracy1_list)
    print('\n')
    print('The average of 10 divisions of accuracy1_score is:', mean_accuracy1)
    print('The max of 10 divisions of accuracy1_score is:', max(accuracy1_list))
    for i in range(len(num_list)):
        mean_accuracy2 = list(map(lambda x: x[i], all_accuracy2_list))
        num_accuracy2_dict[i] = mean_accuracy2
        print('%d Generate the average accuracy2 of 10 divisions under the number: %f' % (
        num_list[i] / 2, np.mean(mean_accuracy2)))
        print('%d Generate the max accuracy2 of 10 divisions under the number: %f' % (
            num_list[i] / 2, max(mean_accuracy2)))
