import sklearn
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# import dalex as dx
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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv(r'./IM_SS_AM1.csv')

X=df.iloc[:,:-1]
le = LabelEncoder()
print(df)
# Encode categorical target in the combined data
# df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
X_true = np.array(df.iloc[0:546, 0:13])
print(X_true)
# print(type(X_true))
y = np.array(df.iloc[0:546, 13])



encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = to_categorical(encoded_Y)
print(y)
test_size = 0.2

X_train, X_val,y_train, y_val = train_test_split(X_true, y, test_size=test_size,random_state=8)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
#standardization

model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,  activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist=model.fit(X_train_scaled, y_train, epochs=250, batch_size=32, verbose=0)
loss, acc = model.evaluate(X_val_scaled, y_val, verbose=0)
# y_val_pred=clf.predict(X_val_scaled)
# y_train_pred = clf.predict(X_train_scaled)
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


deep_explainer  = shap.KernelExplainer(model.predict,X_train_scaled)


#X_te_rs_array = np.asarray(X_te_rs)
shap_values = deep_explainer.shap_values(X_val_scaled,nsamples=10)

class_names = ["AM", "IM", "SS"]
shap.summary_plot(shap_values,X_val_scaled,feature_names=X.columns,class_names=class_names)
# shap.plots._waterfall.waterfall_legacy(deep_explainer.expected_value[0], shap_values[0][0],feature_names=X.columns)
# shap.summary_plot(shap_values[2], X_val_scaled, feature_names = X.columns)
# shap.summary_plot(shap_values[2],X_val_scaled,plot_type="bar",class_names=class_names,feature_names=X.columns, color="purple")
# mean_abs_shap_values_2 = np.mean(np.abs(shap_values[2]), axis=0)
# print(sorted(mean_abs_shap_values_2))

# shap.summary_plot(shap_values[1], X_val_scaled, feature_names = X.columns)
# shap.summary_plot(shap_values[1],X_val_scaled,plot_type="bar",class_names=class_names,feature_names=X.columns, color="purple")
# mean_abs_shap_values_1 = np.mean(np.abs(shap_values[1]), axis=0)
# print(sorted(mean_abs_shap_values_1))
#
shap.summary_plot(shap_values[0], X_val_scaled, feature_names = X.columns)
shap.summary_plot(shap_values[0],X_val_scaled,plot_type="bar",class_names=class_names,feature_names=X.columns, color="purple")
mean_abs_shap_values_0 = np.mean(np.abs(shap_values[0]), axis=0)
print(sorted(mean_abs_shap_values_0))