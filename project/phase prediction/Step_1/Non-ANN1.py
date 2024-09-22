from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.model_selection import KFold
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

df = pd.read_csv(r'./dataset.csv')
le = LabelEncoder()
print(df)
# Encode categorical target in the combined data
df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
X_true = np.array(df.iloc[0:546, :-1])
print((X_true))
y = np.array(df.iloc[0:546, -1])
print(y)
empty_dict = {}


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

y = to_categorical(encoded_Y)
print(y)


test_size = 0.2
accuracy1_list=[]



def create_model():
    model = Sequential()
    model.add(Dense(50,kernel_initializer='normal', input_dim=13, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(60,kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size)
X_train_fea = X_train[:, 0:13]
scaler = StandardScaler()
X_fea = scaler.fit_transform(X_train_fea)
# X_test = scaler.fit_transform(X_test)
X_test_scaled1 = X_test[:, 0:13]
X_test_scaled1 = scaler.fit_transform(X_test_scaled1)
X_test = scaler.fit_transform(X_test)
tensorboard_callback = TensorBoard(log_dir="E:/logic")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_fea, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[tensorboard_callback,early_stopping])
acc = model.evaluate(X_test, y_test, verbose=0)[1]

print(acc)
