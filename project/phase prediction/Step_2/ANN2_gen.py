# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:24:09 2022

@author: swaggy.p
"""
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adamsourceavc
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import os
import optuna
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    '''
    Parameters
    ----------
    X : TYPE: ndarray
        true data
    X_gen : ndarray
        generated data

    Returns
    -------
    X_gen : ndarray
        return generated data with same scale as true data

    '''
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)

    return X_gen

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    def create_bestmodel(trial):
        best_hyperparameters =trial.params
        units1=best_hyperparameters["units1"]
        units2 = best_hyperparameters["units2"]
        dropout_rate1 = best_hyperparameters["dropout_rate1"]
        dropout_rate2 = best_hyperparameters["dropout_rate2"]


        learning_rate=best_hyperparameters["learning_rate"]
        model = Sequential()
        model.add(Dense(units1, kernel_initializer='normal', input_dim=13, activation='relu'))
        model.add(Dropout(dropout_rate1))
        model.add(Dense(units2, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_model(trial):
        units1 = trial.suggest_int('units1', 32, 400, log=True)
        units2 = trial.suggest_int('units2', 32, 400, log=True)
        dropout_rate1 = trial.suggest_float('dropout_rate1', 0.1, 0.8)
        dropout_rate2 = trial.suggest_float('dropout_rate2', 0.1, 0.8)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model = Sequential()
        model.add(Dense(units1, kernel_initializer='normal', input_dim=13, activation='relu'))
        model.add(Dropout(dropout_rate1))
        model.add(Dense(units2, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    def objective(trial):

        units = trial.suggest_int('units', 32, 400, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.8)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

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
            # objection function
            # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            model=create_model(trial)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=100,callbacks=[early_stopping])
            acc = model.evaluate(X_test, y_test, verbose=0)[1]
            accuracy.append(acc)
        return np.mean(accuracy)

        # model=KerasClassifier(build_fn=CNN)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # 设置试验次数

    best_trial = study.best_trial
    best_model = create_bestmodel(best_trial)
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


# -*- coding: utf-8 -*-

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
                              callbacks=[early_stopping])
        accuracy= svr_all.evaluate(X_test,y_test,verbose=0)
        # accuracy2_score = accuracy_score(y_test, y_pred2)
        accuracy2_list.append(accuracy[1])
        # save the generated data
        # gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)),axis=1)
        #gen_data = pd.DataFrame(gen_data)
        # gen_data.to_csv(os.path.join(gen_data_path,'%s_%d_%d.csv'%(random,int(gen_number/2),i)))
    return [int(gen_number / 2), max(accuracy2_list)]
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
if __name__ == '__main__':
    df = pd.read_csv(r'./IM_SS_AM1.csv')
    le = LabelEncoder()
    print(df)
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X_true = np.array(df.iloc[0:546, :-1])
    y = np.array(df.iloc[0:546, -1])
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    y = to_categorical(encoded_Y)
    test_size = 0.2
    num_list = [200, 400, 600, 800, 1000]
    gen_data_path = r'./gen_data_file'
    accuracy1_list = []
    best_number_list = []
    best_rmse2_list = []
    all_accuracy2_list = []
    all_gen_data_list = []
    random_list = list(range(0, 46, 5))

    for random in random_list:
        # divide the dataset based on the current random seed
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size, random_state=random)
        X_train_fea = X_train[:, 0:13]
        scaler = StandardScaler()
        X_fea = scaler.fit_transform(X_train_fea)
        # X_test = scaler.fit_transform(X_test)
        X_test_scaled1 = X_test[:, 0:13]
        X_test_scaled1 = scaler.fit_transform(X_test_scaled1)
        X_test = scaler.fit_transform(X_test)

        svr1 = SVR_opt(X_fea, y_train, use_gendata=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = svr1.fit(X_fea, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
        accuracy= svr1.evaluate(X_test,y_test,verbose=0)
        accuracy1_list.append(accuracy[1])

        # parallel computing on finding different numbers of generated data for each fold of random division
        X_train_scaled = MaxAbsScaler().fit_transform(X_train)
        pool = multiprocessing.Pool(4)
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
        best_rmse2_list.append(max(accuracy2_list))

        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in accuracy2_list:
            if single_rmse2 < accuracy[1]:
                vaild_rmse2.append(single_rmse2)
                index = accuracy2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])

    # integrate the RMSE2 of ten random divisions under each number
    print(accuracy1_list)
    mean_accuracy1 = np.mean(accuracy1_list)
    num_accuracy2_dict = OrderedDict()
    print(accuracy1_list)
    print('\n')
    print('The average of 10 divisions of accuracy1_scoreANN_2gen is:', mean_accuracy1)
    print('The max of 10 divisions of accuracy1_score is:', max(accuracy1_list))
    for i in range(len(num_list)):
        mean_accuracy2 = list(map(lambda x: x[i], all_accuracy2_list))
        num_accuracy2_dict[i] = mean_accuracy2
        print('%d Generate the average accuracy2ANN_2gen of 10 divisions under the number: %f' % (
        num_list[i] / 2, np.mean(mean_accuracy2)))
        print('%d Generate the max accuracy2 of 10 divisions under the number: %f' % (
            num_list[i] / 2, max(mean_accuracy2)))
