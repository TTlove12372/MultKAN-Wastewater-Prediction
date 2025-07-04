import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import joblib
import warnings
import math

# 机器学习和评估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             max_error, explained_variance_score)
from sklearn.ensemble import RandomForestRegressor
from PyEMD import EMD

# TensorFlow/Keras 导入
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau as TFReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal, HeUniform

# 设置随机种子，确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

# 忽略警告
warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = "nn_results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "models"))
    os.makedirs(os.path.join(output_dir, "predictions"))


# 配置日志函数
def log_message(message, file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    if file:
        with open(file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')


# 处理NaN值的函数
def handle_nan(signal):
    """
    处理信号中的NaN值
    """
    # 将无限值转换为NaN
    signal = np.array(signal, dtype=float)
    signal[~np.isfinite(signal)] = np.nan

    # 填充NaN值
    mask = np.isnan(signal)
    if np.all(mask):
        # 如果所有值都是NaN，则用0填充
        return np.zeros_like(signal)

    # 使用前向填充和后向填充处理NaN
    indices = np.arange(len(signal))
    valid_indices = indices[~mask]
    valid_values = signal[~mask]

    # 使用最近的有效值填充
    filled_signal = np.interp(indices, valid_indices, valid_values)

    return filled_signal


# 定义额外的评估指标
def mean_absolute_percentage_error(y_true, y_pred):
    """计算平均绝对百分比误差"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def root_mean_squared_log_error(y_true, y_pred):
    """计算均方根对数误差"""
    # 确保所有值都是正的
    y_true, y_pred = np.maximum(y_true, 0), np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))


def mean_absolute_deviation(y_true, y_pred):
    """计算平均绝对偏差"""
    return np.mean(np.abs(y_true - np.mean(y_true)))


def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name, log_file):
    """评估模型并返回各种指标"""
    # 计算训练集指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    train_rmsle = root_mean_squared_log_error(y_train, y_train_pred)
    train_mad = mean_absolute_deviation(y_train, y_train_pred)
    train_explained_variance = explained_variance_score(y_train, y_train_pred)
    train_max_error = max_error(y_train, y_train_pred)

    # 计算测试集指标
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_rmsle = root_mean_squared_log_error(y_test, y_test_pred)
    test_mad = mean_absolute_deviation(y_test, y_test_pred)
    test_explained_variance = explained_variance_score(y_test, y_test_pred)
    test_max_error = max_error(y_test, y_test_pred)

    # 记录评估结果
    log_message(f"\n===== {model_name} 模型评估结果 =====", log_file)
    log_message(f"训练集评估:", log_file)
    log_message(f"RMSE: {train_rmse:.4f}", log_file)
    log_message(f"MAE: {train_mae:.4f}", log_file)
    log_message(f"R²: {train_r2:.4f}", log_file)
    log_message(f"MAPE: {train_mape:.4f}%", log_file)
    log_message(f"RMSLE: {train_rmsle:.4f}", log_file)
    log_message(f"MAD: {train_mad:.4f}", log_file)
    log_message(f"Explained Variance: {train_explained_variance:.4f}", log_file)
    log_message(f"Max Error: {train_max_error:.4f}", log_file)

    log_message(f"\n测试集评估:", log_file)
    log_message(f"RMSE: {test_rmse:.4f}", log_file)
    log_message(f"MAE: {test_mae:.4f}", log_file)
    log_message(f"R²: {test_r2:.4f}", log_file)
    log_message(f"MAPE: {test_mape:.4f}%", log_file)
    log_message(f"RMSLE: {test_rmsle:.4f}", log_file)
    log_message(f"MAD: {test_mad:.4f}", log_file)
    log_message(f"Explained Variance: {test_explained_variance:.4f}", log_file)
    log_message(f"Max Error: {test_max_error:.4f}", log_file)

    # 创建结果字典
    results = {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'train_rmsle': train_rmsle,
        'train_mad': train_mad,
        'train_explained_variance': train_explained_variance,
        'train_max_error': train_max_error,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'test_rmsle': test_rmsle,
        'test_mad': test_mad,
        'test_explained_variance': test_explained_variance,
        'test_max_error': test_max_error
    }

    return results


def save_predictions(y_test, y_pred, model_name, output_dir):
    """保存测试集预测结果到Excel文件"""
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    pred_df.to_excel(os.path.join(output_dir, "predictions", f"{model_name}_predictions.xlsx"), index=False)


# 添加特征工程和数据预处理函数
def add_lagged_features(X, y, lags=5):
    """添加滞后特征"""
    log_message(f"添加{lags}个滞后特征...")
    for i in range(1, lags + 1):
        y_shifted = y.shift(i) if isinstance(y, pd.Series) else pd.Series(y).shift(i)
        y_shifted = y_shifted.fillna(method='bfill')  # 用后面的值填充NaN
        X[f"y_prev{i}"] = y_shifted

    # 处理X中的任何可能的NaN值
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean() if not pd.isna(X[col].mean()) else 0)

    return X


# ==================== 更新的TensorFlow/Keras 模型定义 ====================


def create_dnn_tf(input_dim):
    """创建简化的深度神经网络模型（TensorFlow/Keras实现）"""
    model = Sequential()

    # 添加第一个隐藏层 - 136个神经元
    model.add(Dense(136, input_dim=input_dim, activation='relu'))

    # 添加输出层
    model.add(Dense(1, activation='linear'))

    # 编译模型 - 无动量优化
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mse', 'mae']
    )

    return model


def create_mlp_tf(input_dim):
    """创建简化的多层感知器模型（TensorFlow/Keras实现）"""
    inputs = Input(shape=(input_dim,))

    # 第一个隐藏层 - 136个神经元
    x = Dense(136, activation='relu')(inputs)

    # 输出层
    outputs = Dense(1, activation='linear')(x)

    # 创建和编译模型 - 无动量优化
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mse', 'mae']
    )

    return model


def train_evaluate_tf_model(model, X_train, y_train, X_test, y_test, model_name,
                            epochs=15, batch_size=32, patience=10, log_file=None):
    """训练和评估TensorFlow/Keras模型"""
    # 配置早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 设置学习率减少策略
    reduce_lr = TFReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    # 开始计时
    start_time = time.time()

    # 训练模型 - 已将epochs降至50
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # 使用20%的训练数据作为验证集
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 训练时间
    training_time = time.time() - start_time
    log_message(f"{model_name} 训练时间: {training_time:.2f} 秒", log_file)

    # 保存模型
    model.save(os.path.join(output_dir, "models", f"{model_name}.h5"))

    # 预测
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    # 保存预测结果
    save_predictions(y_test, y_test_pred, model_name, output_dir)

    # 评估模型
    results = evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name, log_file)

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    # 绘制真实值与预测值的散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'{model_name} 预测 vs 真实值')
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_history.png"), dpi=300)
    plt.close()

    return results, history, y_train_pred, y_test_pred


def load_saved_indices(indices_file='dataset_indices.npz'):
    """加载预先保存的训练/测试集索引"""
    try:
        loaded_indices = np.load(indices_file)
        train_indices = loaded_indices['train_indices']
        val_indices = loaded_indices['val_indices']
        return train_indices, val_indices
    except Exception as e:
        return None, None


def main():
    # 创建日志文件
    log_file = os.path.join(output_dir, "training_log.txt")

    # 开始计时
    start_time = time.time()

    try:
        # 读取数据
        log_message("开始读取数据...", log_file)
        try:
            # 替换为实际的文件路径
            file_path = r"C:\Users\肖亚宁\PycharmProjects\pythonProjecttensor\最终\平定6指标_cleaned\平定6指标_cleaned\cleaned_平定中伟—TP.xlsx"
            data = pd.read_excel(file_path)
            log_message(f"成功读取数据，形状: {data.shape}", log_file)

            # 显示数据的基本信息
            log_message(f"数据列: {data.columns.tolist()}", log_file)
            log_message(f"数据类型:\n{data.dtypes}", log_file)
            log_message(f"数据统计摘要:\n{data.describe()}", log_file)

            # 检查缺失值
            missing_values = data.isnull().sum()
            log_message(f"缺失值统计:\n{missing_values}", log_file)

            if missing_values.sum() > 0:
                log_message("发现缺失值，进行处理...", log_file)
                # 对数值列使用中位数填充缺失值
                numeric_cols = data.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if data[col].isnull().sum() > 0:
                        median = data[col].median()
                        data[col].fillna(median, inplace=True)
                        log_message(f"列 '{col}' 的缺失值已用中位数 {median} 填充", log_file)

            # 分离特征和目标变量
            X = data.iloc[:, :-1]  # 所有列除了最后一列
            y = data.iloc[:, -1]  # 最后一列

            log_message(f"原始特征数量: {X.shape[1]}", log_file)
            log_message(f"目标变量名称: {data.columns[-1]}", log_file)

            # ==================== 添加特征工程 ====================
            # 添加前5天的滞后特征
            X = add_lagged_features(X, y, lags=5)
            log_message(f"添加滞后特征后的特征数量: {X.shape[1]}", log_file)

            # 直接使用添加了滞后特征的数据
            X_for_ml = X.values
            y_filtered = y.values

            # 加载预先保存的索引
            log_message("尝试加载预先保存的训练/测试集索引...", log_file)
            train_indices, test_indices = load_saved_indices('dataset_indices.npz')

            if train_indices is not None and test_indices is not None:
                log_message(f"成功加载索引。训练集: {len(train_indices)}个样本, 测试集: {len(test_indices)}个样本",
                            log_file)

                # 使用加载的索引分割数据
                X_train = X_for_ml[train_indices]
                X_test = X_for_ml[test_indices]
                y_train = y_filtered[train_indices]
                y_test = y_filtered[test_indices]

                log_message(f"使用保存的索引划分数据。训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本",
                            log_file)
            else:
                log_message("未找到或无法加载保存的索引，回退到随机分割", log_file)
                # 如果没有找到保存的索引，则使用传统的随机分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X_for_ml, y_filtered, test_size=0.2, random_state=42)
                log_message(f"随机分割数据。训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本", log_file)

            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 保存缩放器以便将来使用
            joblib.dump(scaler, os.path.join(output_dir, "models", "scaler.pkl"))

            # 设置GPU配置
            # TensorFlow GPU 配置
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # 设置内存增长而不是一次性分配所有内存
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    log_message(f"TensorFlow 检测到 {len(gpus)} 个 GPU 设备", log_file)
                except RuntimeError as e:
                    log_message(f"TensorFlow GPU 配置错误: {e}", log_file)
            else:
                log_message("TensorFlow 未检测到 GPU 设备", log_file)

            # 保存所有模型评估结果的列表
            all_results = []

            # 输入维度
            input_dim = X_train_scaled.shape[1]


            # 3. 深度神经网络 (DNN) - TensorFlow
            log_message("\n开始训练 DNN TensorFlow 模型...", log_file)
            dnn_tf_model = create_dnn_tf(input_dim=input_dim)

            dnn_tf_results, dnn_tf_history, dnn_tf_train_pred, dnn_tf_test_pred = train_evaluate_tf_model(
                dnn_tf_model, X_train_scaled, y_train, X_test_scaled, y_test,
                "DNN_TensorFlow", epochs=50, batch_size=32, patience=30, log_file=log_file
            )
            all_results.append(dnn_tf_results)

            # 4. 多层感知器 (MLP) - TensorFlow
            log_message("\n开始训练 MLP TensorFlow 模型...", log_file)
            mlp_tf_model = create_mlp_tf(input_dim=input_dim)

            mlp_tf_results, mlp_tf_history, mlp_tf_train_pred, mlp_tf_test_pred = train_evaluate_tf_model(
                mlp_tf_model, X_train_scaled, y_train, X_test_scaled, y_test,
                "MLP_TensorFlow", epochs=50, batch_size=32, patience=30, log_file=log_file
            )
            all_results.append(mlp_tf_results)

            # 创建汇总结果DataFrame
            results_df = pd.DataFrame(all_results)

            # 保存结果到Excel
            results_df.to_excel(os.path.join(output_dir, "model_evaluation_results.xlsx"), index=False)

            # 绘制模型性能对比图
            metrics = ['test_rmse', 'test_r2', 'test_mae', 'test_mape']
            plt.figure(figsize=(16, 10))

            for i, metric in enumerate(metrics):
                plt.subplot(2, 2, i + 1)
                sns.barplot(x='model_name', y=metric, data=results_df)
                plt.title(f'模型 {metric} 对比')
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)

            # 计算总运行时间
            end_time = time.time()
            total_time = end_time - start_time
            log_message(f"\n总运行时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)", log_file)

            log_message("\n训练与评估完成！所有结果已保存到 " + output_dir, log_file)

        except Exception as e:
            log_message(f"发生错误: {str(e)}", log_file)
            import traceback
            log_message(traceback.format_exc(), log_file)

    except Exception as e:
        print(f"出现错误: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()