import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVR,SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd
from collections import defaultdict,OrderedDict
from sklearn.model_selection import KFold, train_test_split
import multiprocessing
from functools import partial
import pickle
import warnings
from sklearn.metrics import classification_report
import os
from sklearn.preprocessing import LabelEncoder
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


            idx = np.random.randint(0, X_data.shape[0], half_batch)
            imgs = X_data[idx]

            noise = np.random.normal(0, 1, (half_batch, 10))

            gen_imgs = self.generator.predict(noise)

            # calculate discriminator loss
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            noise = np.random.normal(0, 1, (batch_size, 10))

            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # save data at certain intervals
            if epoch % save_interval == 0:
                noise = np.random.normal(0, 1, (batch_size, 10))
                gen_imgs = self.generator.predict(noise)

        self.gen_data = gen_imgs


def Inverse_transform(X, X_gen):

    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)

    return X_gen
from sklearn.ensemble import RandomForestClassifier
def SVR_opt(X_true, y_true, X_gen=None, y_gen=None, use_gendata=True):
    def  objective (trial):
        param = {

            "max_depth": trial.suggest_int("max_depth", 1, 32, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),

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

        gan = GAN()
        gan.train(epochs=800, X_data=X_train_scaled, batch_size=gen_number, save_interval=50)
        gen_data = gan.gen_data


        X_gen_fea, y_gen, X_gen_fea_origin, X_fea = Target_predict(X_train, y_train, gen_data)


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

        class_report = classification_report(y_test, y_pred2, output_dict=True)
        log_message = f"random: {random}, i: {i}, gen_number: {gen_number}, overall accuracy: {accuracy2_score}"
        print(log_message)

        for label, metrics in class_report.items():
            if isinstance(metrics, dict):  # Filter to avoid 'accuracy' key, which is not a label
                class_accuracy = metrics['precision']
                class_log_message = f"class {label}: accuracy (precision) = {class_accuracy}"
                print(class_log_message)

                with open("accuracy_RF_log.txt", "a") as f:
                    f.write(log_message + "\n" + class_log_message + "\n")

        gen_data = np.concatenate((X_gen_fea_origin, y_gen.reshape(-1,1)),axis=1)
        gen_data = pd.DataFrame(gen_data)
        gen_data.to_csv(os.path.join(gen_data_path,'%s_%d_%d.csv'%(random,int(gen_number/2),i)))

    return [int(gen_number / 2), max(accuracy2_list)]


# Function to load and prepare data
def load_data(file_path):
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
    X = np.array(df.iloc[0:546, 0:13])  # Features (first 13 columns)
    y = np.array(df.iloc[0:546, 13])  # Target (13th column)
    return X, y

# Function to scale features using StandardScaler
def scale_features(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to scale features using MaxAbsScaler
def scale_maxabs(X_train, X_test):
    scaler = MaxAbsScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to calculate accuracy score
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Function to handle SVR optimization
def optimize_svr(X_train, y_train, use_gendata=False):
    svr_model = SVR_opt(X_train, y_train, use_gendata=use_gendata)
    svr_model.fit(X_train, y_train)
    return svr_model

# Function to generate data using random search
def perform_random_search(random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled, num_list):
    pool = multiprocessing.Pool(5)
    results = pool.map(
        partial(random_search, random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled), num_list)
    pool.close()
    pool.join()
    return results

# Main function
def main():
    # Load and prepare data
    df_path = './IM_SS_AM1.csv'
    X_true, y = load_data(df_path)

    # Initialize variables
    test_size = 0.2
    num_list = [200, 400, 600, 800, 1000]
    gen_data_path = './gen_RF_data_file'
    os.makedirs(gen_data_path, exist_ok=True)

    accuracy1_list = []
    best_number_list = []
    best_rmse2_list = []
    all_accuracy2_list = []
    random_list = list(range(0, 46, 5))

    # Loop through random splits for cross-validation
    for random in random_list:
        X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size, random_state=random)

        # Feature scaling for SVR model
        X_train_scaled, X_test_scaled = scale_features(X_train[:, 0:13], X_test[:, 0:13])

        # Fit SVR model and evaluate accuracy
        svr_model = optimize_svr(X_train_scaled, y_train, use_gendata=False)
        y_pred1 = svr_model.predict(X_test_scaled)
        accuracy1_score = calculate_accuracy(y_test, y_pred1)
        accuracy1_list.append(accuracy1_score)

        # Scale features for random search
        X_train_scaled_maxabs, X_test_scaled_maxabs = scale_maxabs(X_train, X_test)

        # Perform random search for data generation
        results = perform_random_search(random, gen_data_path, X_train, y_train, X_test, y_test, X_train_scaled_maxabs, num_list)

        # Extract results
        gen_num = [x[0] for x in results]
        accuracy2_list = list(map(lambda x: x[1], results))
        all_accuracy2_list.append(accuracy2_list)

        # Get the best performing number of generated samples
        best_idx = accuracy2_list.index(max(accuracy2_list))
        best_number_list.append(gen_num[best_idx])
        best_rmse2_list.append(max(accuracy2_list))

        # Filter results where accuracy2 < accuracy1_score
        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in accuracy2_list:
            if single_rmse2 < accuracy1_score:
                vaild_rmse2.append(single_rmse2)
                index = accuracy2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])

    # Calculate and print results
    mean_accuracy1 = np.mean(accuracy1_list)
    print("Accuracy results for 10 splits:")
    print(accuracy1_list)
    print(f"\nThe average accuracy1_score over 10 splits is: {mean_accuracy1}")
    print(f"The maximum accuracy1_score over 10 splits is: {max(accuracy1_list)}")

    num_accuracy2_dict = OrderedDict()
    for i, num in enumerate(num_list):
        mean_accuracy2 = [x[i] for x in all_accuracy2_list]
        num_accuracy2_dict[i] = mean_accuracy2
        print(f"{num // 2} Generated: average accuracy2 = {np.mean(mean_accuracy2)}")
        print(f"{num // 2} Generated: max accuracy2 = {max(mean_accuracy2)}")

if __name__ == '__main__':
    main()