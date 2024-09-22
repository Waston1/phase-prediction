# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:24:09 2022

@author: swaggy.p
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.model_selection import KFold
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
# gpus = tf.config.list_physical_devices(device_type='GPU')
# cpus = tf.config.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')
for gpu in gpus[0:2]:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# plot rodar figures
def plot_data(datas, number):
    features = ['δr', '∆χ', 'VEC', 'delta H', '∆S', 'Ω', 'Λ', 'γ parameter', 'D.χ', 'e1/a', 'e2/a', 'Ec',
                'η', 'D.r', 'A', 'F', 'w', 'G', 'δG', 'D.G', 'μ', 'Hardness']
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    features = np.concatenate((features, [features[0]]))

    plt.figure(figsize=(6, 6), facecolor='white')
    plt.subplot(111, polar=True)
    for value in datas:
        value = np.concatenate((value, [value[0]]))
        # plt.subplot(111,polar=True)
        plt.polar(angles, value, 'b-', linewidth=1, alpha=0.2)
        plt.fill(angles, value, alpha=0.25, color='g')
        plt.thetagrids(angles * 180 / np.pi, features)

    plt.grid()
    # plt.savefig()


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
            d_loss_real = self.discriminator.train_on_batch(imgs, 0.9*np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,0.1*np.ones((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #   triain generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise,  0.9 * np.ones((batch_size, 1)))

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

from sklearn.metrics import accuracy_score
def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    # objection function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def clf_validation(trial):
        C=trial.suggest_loguniform("C",0.0001, 10000)
        gamma =trial.suggest_loguniform('gamma',0.0001, 1)

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
            SVR_=SVC(gamma=gamma,C=C).fit(X_train,y_train)
            y_pred = SVR_.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
        return np.mean(accuracy)

    study = optuna.create_study(direction="maximize")
    study.optimize(clf_validation, n_trials=50)
    trial = study.best_trial
    best_hyperparameters = trial.params
    svr = SVC(gamma= best_hyperparameters["gamma"],
              C=best_hyperparameters['C'])
    return svr

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


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:58:39 2022

@author: C903
"""
import os
import numpy as np
import matplotlib as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import optuna
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import multiprocessing
from functools import partial
import pickle
import warnings

warnings.filterwarnings('ignore')
from collections import OrderedDict


# Bayesian tuning hyperparameters


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
        # save the generated data
        gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)),axis=1)
        gen_data = pd.DataFrame(gen_data)
        # gen_data.to_csv(os.path.join(gen_data_path,'%s_%d_%d.csv'%(random,int(gen_number/2),i)))
    return [int(gen_number / 2), max(accuracy2_list)]

from sklearn.preprocessing import LabelEncoder
if __name__ == '__main__':

    # import dataset
    df = pd.read_csv(r'./IM_SS_AM1.csv')
    le = LabelEncoder()
    print(df)
    # Encode categorical target in the combined data
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X_true = np.array(df.iloc[0:546, 0:13])
    # print(type(X_true))
    y = np.array(df.iloc[0:546, 13])
    test_size = 0.2

    # half batch =[100, 200, 300, 400, 500] = actual geneerated number
    num_list = [200, 400, 600, 800, 1000]

    gen_data_path = r'./gen_data_file'
    accuracy1_list = []
    best_number_list = []
    best_rmse2_list = []
    all_accuracy2_list = []
    all_gen_data_list = []
    best_accuracy2_list=[]
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

        # parallel computing on finding different numbers of generated data for each fold of random division
        X_train_scaled = MaxAbsScaler().fit_transform(X_train)
        pool = multiprocessing.Pool(5)
        results = pool.map(
            partial(random_search, random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled), num_list)
        pool.close()
        pool.join()

        # integrate the model predictiorn results of different numbers of generation under current division and compare with accuracy1_score
        gen_num = [x[0] for x in results]
        accuracy2_list = list(map(lambda x: x[1], results))
        all_accuracy2_list.append(accuracy2_list)

        # find the optimal number of generations under each division

        ind = accuracy2_list.index(max(accuracy2_list))
        best_number_list.append(gen_num[ind])
        best_accuracy2_list.append(max(accuracy2_list))

        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in accuracy2_list:
            if single_rmse2 < accuracy1_score:
                vaild_rmse2.append(single_rmse2)
                index = accuracy2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])

    # integrate the RMSE2 of ten random divisions under each number
    # print(max_accuracy1_list=max(accuracy1_list))
    mean_accuracy1 = np.mean(accuracy1_list)
    num_accuracy2_dict = OrderedDict()
    print(accuracy1_list)
    print('\n')
    print('The average of 10 divisions of accuracy1_score is:', mean_accuracy1)
    print('The average of 10 divisions of accuracy1_score is:', max(accuracy1_list))
    for i in range(len(num_list)):
        mean_accuracy2 = list(map(lambda x: x[i], all_accuracy2_list))
        num_accuracy2_dict[i] = mean_accuracy2

        print('%d Generate the average accuracy2 of 10 divisions under the number: %f' % (
        num_list[i] / 2, np.mean(mean_accuracy2)))
        print('%d Generate the max accuracy2 of 10 divisions under the number: %f' % (
            num_list[i] / 2, max(mean_accuracy2)))