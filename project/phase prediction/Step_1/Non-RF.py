from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # 导入SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'./dataset.csv')
le = LabelEncoder()
print(df)
# Encode categorical target in the combined data
df['Phase_inshort'] = le.fit_transform(df['Phase_inshort'].astype(str))
X_true = np.array(df.iloc[0:546, 0:13])
print(X_true)
# print(type(X_true))
y = np.array(df.iloc[0:546, 13])
print(y)
test_size = 0.2
accuracy1_list=[]

random_list = list(range(0, 46, 5))
for random in random_list:
    X_train, X_test, y_train, y_test = train_test_split(X_true, y, test_size=test_size,random_state=random)
    X_train_fea = X_train[:, 0:13]
    scaler1 = StandardScaler().fit(X_train_fea)
    X_fea = scaler1.transform(X_train_fea)
    X_test_scaled1 = X_test[:, 0:13]
    X_test_scaled1 = scaler1.transform(X_test_scaled1)

    clf = RandomForestClassifier()
    clf.fit(X_fea,y_train)
    y_pred1 = clf.predict(X_test_scaled1)
    accuracy1_score=accuracy_score(y_test, y_pred1)
    accuracy1_list.append(accuracy1_score)

print(accuracy1_list)
mean_accurary=np.mean(accuracy1_list)
print(mean_accurary)

