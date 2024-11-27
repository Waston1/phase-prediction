import pandas as pd
import joblib

# 加载模型和标准化器
model_path = 'best_model_svr.joblib'
scaler_path = 'best_scaler_svr.joblib'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 加载验证数据
data_path = 'valida2.csv'
data = pd.read_csv(data_path)

# 假设数据的最后一列是标签
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 对特征进行标准化
X_scaled = scaler.transform(X)

# 进行预测
predictions = model.predict(X_scaled)

# 保存预测结果到CSV文件
output = pd.DataFrame({
    'True_Label': y,
    'Predicted_Label': predictions
})
output.to_csv('predictions.csv', index=False)

print("预测已完成，结果已保存至 predictions.csv")
