import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error
from kan.MultKAN import MultKAN


def preprocess_excel_data(file_path):
    """
    Excel数据预处理函数，处理多小数点错误、空值和NaN

    参数:
    file_path (str): Excel文件路径

    返回:
    pd.DataFrame: 预处理后的数据框
    """
    import re

    # 读取Excel文件
    data = pd.read_excel(file_path)

    # 处理每一列
    for col in data.columns:
        # 判断是否为日期列
        is_date_col = any(date_term in str(col).lower()
                          for date_term in ['date', '日期', 'time', '时间'])

        # 对非日期列进行数值处理
        if not is_date_col:
            # 修复字符串格式的数值问题
            if data[col].dtype == object:
                def fix_multiple_decimals(x):
                    if not isinstance(x, str):
                        return x

                    # 空字符串处理
                    if x.strip() == '':
                        return np.nan

                    # 首先替换连续的小数点 (如 '4..4' → '4.4')
                    x = re.sub(r'\.+', '.', x)

                    # 如果仍有多个小数点，保留第一个，删除其余 (如 '12.34.56' → '12.3456')
                    if x.count('.') > 1:
                        first_dot = x.find('.')
                        x = x[:first_dot + 1] + x[first_dot + 1:].replace('.', '')

                    # 删除所有非数字和非小数点字符
                    x = ''.join(c for c in x if c.isdigit() or c == '.')

                    return x

                data[col] = data[col].apply(fix_multiple_decimals)

            # 转换为数值，错误转为NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')

            # 处理NaN：先用插值，再用均值
            if data[col].isna().any():
                data[col] = data[col].interpolate(method='linear', limit_direction='both')
                if data[col].isna().any():
                    mean_val = data[col].mean()
                    # 如果均值是NaN（所有值都是NaN），则用0填充
                    data[col] = data[col].fillna(0 if pd.isna(mean_val) else mean_val)

    # 按日期排序（如果有日期列）
    date_cols = [col for col in data.columns if any(date_term in str(col).lower()
                                                    for date_term in ['date', '日期', 'time', '时间'])]
    if date_cols:
        data = data.sort_values(by=date_cols[0])

    return data


# 计算所有指标的函数
def calculate_metrics(actual, pred):
    """计算所有指标"""
    # 确保数据形状正确
    actual = actual.flatten()
    pred = pred.flatten()

    # 基本指标
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    # MAPE (Mean Absolute Percentage Error)
    mask = np.abs(actual) > 1e-10
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if any(mask) else np.nan

    # RMSLE (Root Mean Squared Logarithmic Error)
    # 处理非正值
    actual_pos = np.maximum(actual, 1e-10)
    pred_pos = np.maximum(pred, 1e-10)
    rmsle = np.sqrt(np.mean(np.power(np.log1p(actual_pos) - np.log1p(pred_pos), 2)))

    # MAD (Mean Absolute Deviation)
    mad = np.mean(np.abs(actual - np.mean(actual)))

    # EVS (Explained Variance Score)
    evs = explained_variance_score(actual, pred)

    # Max_Error
    max_err = max_error(actual, pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'RMSLE': rmsle,
        'MAD': mad,
        'EVS': evs,
        'Max_Error': max_err
    }


# 设置设备 - 添加GPU支持
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据加载与预处理
file_path = r"C:\Users\肖亚宁\PycharmProjects\pythonProjecttensor\最终\平定6指标_cleaned\平定6指标_cleaned\cleaned_平定中伟—TP.xlsx"

# 使用新的预处理函数
data = preprocess_excel_data(file_path)

# 确保数据按时间排序
if 'date' in data.columns or '日期' in data.columns:
    date_col = 'date' if 'date' in data.columns else '日期'
    print(f"数据已按{date_col}排序")
else:
    print("警告：未找到日期列，假设数据已按时间顺序排列")

# 特征与目标变量分离
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]  # 最后一列为目标变量

# 添加前7天的输出变量作为特征
print("添加前5天的输出变量作为特征...")
for i in range(1, 6):  # 前5天
    y_shifted = y.shift(i)
    y_shifted = y_shifted.fillna(method='bfill')  # 用后面的值填充NaN
    X[f"y_prev{i}"] = y_shifted

# 处理X中的任何可能的NaN值
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].mean() if not pd.isna(X[col].mean()) else 0)

print(f"特征数: {X.shape[1]}")
print(f"特征列表: {list(X.columns)}")

# 准备MultKAN模型的数据
X_scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = X_scaler.fit_transform(X)

y_array = y.values.reshape(-1, 1)
y_scaler = StandardScaler()
y_normalized = y_scaler.fit_transform(y_array)

# 转换为PyTorch张量并移动到GPU
x_tensor = torch.tensor(X_normalized, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32, device=device)

# 设置随机种子以确保可重复性
torch.manual_seed(44)  # 固定随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

# 划分训练集和验证集（80%/20%）
train_size = int(0.8 * len(x_tensor))
indices = torch.randperm(len(x_tensor), device=device)  # 在GPU上生成随机索引
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# ------------------- 保存训练集和验证集索引 -------------------
# 创建要保存的字典
indices_dict = {
    'train_indices': train_indices.cpu().numpy(),
    'val_indices': val_indices.cpu().numpy()
}

# 保存索引到文件
save_path = 'dataset_indices_no_rf_emd.npz'
np.savez(save_path, **indices_dict)
print(f"数据集索引已保存到: {save_path}")

# 如果需要在其他代码中加载这些索引，可以使用以下代码：
# loaded_indices = np.load('dataset_indices_no_rf_emd.npz')
# train_indices = torch.tensor(loaded_indices['train_indices'], device=device)
# val_indices = torch.tensor(loaded_indices['val_indices'], device=device)
# --------------------------------------------------------------

x_train = x_tensor[train_indices]
y_train = y_tensor[train_indices]
x_val = x_tensor[val_indices]
y_val = y_tensor[val_indices]

# 配置MultKAN模型
input_dim = X.shape[1]
hidden_layers = [72, 64]
output_dim = 1
grid_size = 9
k_value = 3
lambda_value = 0.02
optimizer = 'LBFGS'
training_steps = 15
print(f"MultKAN模型输入维度: {input_dim}")

# 创建MultKAN模型并移动到GPU
model = MultKAN(width=[input_dim, hidden_layers, output_dim], grid=grid_size, k=k_value, seed=1).to(device)

dataset = {
    'train_input': x_train,
    'train_label': y_train,
    'test_input': x_val,
    'test_label': y_val
}

# 训练模型
model.fit(dataset, opt=optimizer, steps=training_steps, lamb=lambda_value)

# 评估模型性能
model.eval()
with torch.no_grad():
    train_pred_norm = model(x_train)
    val_pred_norm = model(x_val)

# 逆转换回原始尺度进行评估
train_pred = y_scaler.inverse_transform(train_pred_norm.cpu().numpy())
val_pred = y_scaler.inverse_transform(val_pred_norm.cpu().numpy())
train_actual = y_scaler.inverse_transform(y_train.cpu().numpy())
val_actual = y_scaler.inverse_transform(y_val.cpu().numpy())

# 计算所有评估指标
train_metrics = calculate_metrics(train_actual, train_pred)
val_metrics = calculate_metrics(val_actual, val_pred)

# 打印训练集和验证集的所有指标
print("\n训练集指标:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n验证集指标:")
for metric, value in val_metrics.items():
    print(f"{metric}: {value:.4f}")

# 打印模型的具体参数
print("\n模型参数:")
print(f"输入维度: {input_dim}")
print(f"隐藏层维度: {hidden_layers}")
print(f"输出维度: {output_dim}")
print(f"网格数量 (grid): {grid_size}")
print(f"核心数量 (k): {k_value}")
print(f"正则化参数 (lamb): {lambda_value}")
print(f"优化器: {optimizer}")
print(f"训练步数: {training_steps}")
print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")

# 详细打印每层的参数大小
parameter_sizes = {}
for name, param in model.named_parameters():
    parameter_sizes[name] = str(param.size())
    print(f"{name}: {param.size()}")

# 保存模型参数和指标到一个Excel文件
model_params_file = 'MultKAN模型参数_无RF_EMD.xlsx'
with pd.ExcelWriter(model_params_file, engine='openpyxl') as writer:
    # 创建模型参数DataFrame并保存
    model_params = {
        '参数名称': [
            '输入维度', '隐藏层维度', '输出维度', '网格数量 (grid)',
            '核心数量 (k)', '正则化参数 (lamb)', '优化器', '训练步数',
            '总参数数量'
        ],
        '参数值': [
            input_dim, str(hidden_layers), output_dim, grid_size,
            k_value, lambda_value, optimizer, training_steps,
            sum(p.numel() for p in model.parameters())
        ]
    }

    params_df = pd.DataFrame(model_params)
    params_df.to_excel(writer, sheet_name='模型参数', index=False)

    # 保存详细的层参数大小
    layer_params = {
        '层名称': list(parameter_sizes.keys()),
        '参数大小': list(parameter_sizes.values())
    }
    layer_df = pd.DataFrame(layer_params)
    layer_df.to_excel(writer, sheet_name='层参数详情', index=False)

    # 保存训练集指标
    train_metrics_df = pd.DataFrame({
        '指标名称': list(train_metrics.keys()),
        '训练集值': [f"{v:.4f}" for v in train_metrics.values()]
    })
    train_metrics_df.to_excel(writer, sheet_name='训练集指标', index=False)

    # 保存验证集指标
    val_metrics_df = pd.DataFrame({
        '指标名称': list(val_metrics.keys()),
        '验证集值': [f"{v:.4f}" for v in val_metrics.values()]
    })
    val_metrics_df.to_excel(writer, sheet_name='验证集指标', index=False)

print(f"\n模型参数和指标已保存到 {model_params_file}")

# 保存测试集预测结果到另一个Excel文件
results_file = 'MultKAN预测结果_无RF_EMD.xlsx'
results_df = pd.DataFrame()

# 添加日期列(如果存在)
if 'date' in data.columns or '日期' in data.columns:
    date_col = 'date' if 'date' in data.columns else '日期'
    original_val_indices = val_indices.cpu().numpy()
    results_df['日期'] = data[date_col].iloc[original_val_indices].reset_index(drop=True)

# 添加真实值和预测值
results_df['真实值'] = val_actual.flatten()
results_df['预测值'] = val_pred.flatten()
results_df['绝对误差'] = np.abs(val_actual.flatten() - val_pred.flatten())
results_df['相对误差(%)'] = np.abs((val_actual.flatten() - val_pred.flatten()) / val_actual.flatten()) * 100

# 保存到Excel文件
results_df.to_excel(results_file, index=False)
print(f"预测结果已保存到 {results_file}")

# 可视化验证集预测结果
plt.figure(figsize=(10, 6))
plt.scatter(val_actual, val_pred, alpha=0.7)
plt.plot([min(val_actual), max(val_actual)], [min(val_actual), max(val_actual)], 'r--')
plt.xlabel('实际出水COD (mg/L)')
plt.ylabel('预测出水COD (mg/L)')
plt.title('MultKAN模型预测结果（无RF和EMD）')
plt.grid(True)
plt.savefig('prediction_results_no_rf_emd.png')
plt.show()

# 可视化预测的时间序列
if 'date' in data.columns or '日期' in data.columns:
    date_col = 'date' if 'date' in data.columns else '日期'
    # 获取验证集的日期
    original_val_indices = val_indices.cpu().numpy()
    val_dates = data[date_col].iloc[original_val_indices]

    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(val_dates), val_actual, 'b-', label='实际值')
    plt.plot(pd.to_datetime(val_dates), val_pred, 'r--', label='预测值')
    plt.xlabel('日期')
    plt.ylabel('出水COD (mg/L)')
    plt.title('COD时间序列预测（无RF和EMD）')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_series_no_rf_emd.png')
    plt.show()
else:
    # 如果没有日期列，使用样本索引
    plt.figure(figsize=(14, 7))
    plt.plot(val_actual, 'b-', label='实际值')
    plt.plot(val_pred, 'r--', label='预测值')
    plt.xlabel('样本索引')
    plt.ylabel('出水COD (mg/L)')
    plt.title('COD时间序列预测（无RF和EMD）')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_series_no_rf_emd.png')
    plt.show()