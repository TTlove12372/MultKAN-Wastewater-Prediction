import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor  # For AdaBoost base estimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, explained_variance_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from PyEMD import EMD  # Added for EMD decomposition
import warnings
import time
import joblib
import math
import tempfile

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可重复
np.random.seed(44)

# 强制joblib使用threading后端而不是multiprocessing，避免Unicode问题
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# 创建输出目录
output_dir = "results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "models"))
    os.makedirs(os.path.join(output_dir, "predictions"))


# 处理NaN值的函数
def handle_nan(signal):
    """处理信号中的NaN值"""
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


# 配置日志函数
def log_message(message, file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    if file:
        # 使用UTF-8编码打开文件，解决编码问题
        with open(file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')


# 定义MAPE和RMSLE函数
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


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, log_file):
    """评估模型并返回各种指标"""
    # 训练集预测
    y_train_pred = model.predict(X_train)

    # 测试集预测
    y_test_pred = model.predict(X_test)

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
    # 避免使用特殊字符R²，改为用R平方
    log_message(f"R平方: {train_r2:.4f}", log_file)
    log_message(f"MAPE: {train_mape:.4f}%", log_file)
    log_message(f"RMSLE: {train_rmsle:.4f}", log_file)
    log_message(f"MAD: {train_mad:.4f}", log_file)
    log_message(f"Explained Variance: {train_explained_variance:.4f}", log_file)
    log_message(f"Max Error: {train_max_error:.4f}", log_file)

    log_message(f"\n测试集评估:", log_file)
    log_message(f"RMSE: {test_rmse:.4f}", log_file)
    log_message(f"MAE: {test_mae:.4f}", log_file)
    log_message(f"R平方: {test_r2:.4f}", log_file)
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

    # 保存测试集预测结果
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred
    })
    pred_df.to_excel(os.path.join(output_dir, "predictions", f"{model_name}_predictions.xlsx"), index=False)

    # 返回结果和预测值
    return results, y_train_pred, y_test_pred


def detect_and_remove_outliers(X, y, method='IQR', threshold=1.5):
    """检测并移除异常值"""
    log_message(f"原始数据形状: {X.shape}, {y.shape}")

    if method == 'IQR':
        log_message(f"未应用异常值检测，使用原始数据")

        return X_cleaned, y_cleaned
    else:
        log_message(f"未应用异常值检测，使用原始数据")
        return X, y


def load_saved_indices(indices_file='dataset_indices.npz'):
    """加载预先保存的训练/测试集索引"""
    try:
        loaded_indices = np.load(indices_file)
        train_indices = loaded_indices['train_indices']
        val_indices = loaded_indices['val_indices']
        return train_indices, val_indices
    except Exception as e:
        print(f"加载索引文件失败: {e}")
        return None, None


# 添加特征工程函数
def add_lagged_features(X, y, log_file=None, lags=5):
    """添加滞后特征"""
    log_message(f"添加前{lags}天的输出变量作为特征...", log_file)
    X_with_lags = X.copy()

    for i in range(1, lags + 1):
        y_shifted = y.shift(i) if isinstance(y, pd.Series) else pd.Series(y).shift(i)
        y_shifted = y_shifted.fillna(method='bfill')  # 用后面的值填充NaN
        X_with_lags[f"y_prev{i}"] = y_shifted

    # 处理X中的任何可能的NaN值
    for col in X_with_lags.columns:
        if X_with_lags[col].isna().any():
            col_mean = X_with_lags[col].mean()
            if pd.isna(col_mean):
                X_with_lags[col] = X_with_lags[col].fillna(0)
            else:
                X_with_lags[col] = X_with_lags[col].fillna(col_mean)

    log_message(f"添加滞后特征后的特征数量: {X_with_lags.shape[1]}", log_file)
    return X_with_lags


def select_features_with_rf(X, y, top_n=10, log_file=None):
    """使用随机森林进行特征选择"""
    log_message(f"开始使用随机森林进行特征选择...", log_file)

    # 创建并训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # 使用所有可用CPU
    )

    rf_model.fit(X, y)

    # 获取特征重要性
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 打印特征重要性
    log_message("特征重要性排名:", log_file)
    log_message(str(feature_importances.head(10)), log_file)  # 显示前10个最重要的特征

    # 选择前N个特征
    selected_features = feature_importances.head(top_n)['feature'].tolist()
    log_message(f"选中了{len(selected_features)}个特征用于模型训练:", log_file)
    for f in selected_features:
        log_message(f"- {f}", log_file)

    # 仅保留选中的特征
    X_selected = X[selected_features]

    return X_selected, selected_features


def main():
    # 创建日志文件
    log_file = os.path.join(output_dir, "training_log.txt")

    # 开始计时
    start_time = time.time()

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

        log_message(f"特征数量: {X.shape[1]}", log_file)
        log_message(f"目标变量名称: {data.columns[-1]}", log_file)

        # ==================== 简化的特征工程部分 ====================
        # 1. 添加滞后特征
        X = add_lagged_features(X, y, log_file, lags=5)

        # 2. 使用随机森林选择最重要的特征
        X_selected, selected_features = select_features_with_rf(X, y, top_n=3, log_file=log_file)

        # 使用选择后的特征
        X_processed = X_selected.values

        # 检测并移除异常值 - 改为不处理异常值以保持数据一致性
        X_processed, y_values = detect_and_remove_outliers(X_processed, y.values, method='none')

        # 尝试加载预先保存的训练/测试集索引
        log_message("尝试加载预先保存的训练/测试集索引...", log_file)
        train_indices, test_indices = load_saved_indices('dataset_indices.npz')

        if train_indices is not None and test_indices is not None:
            # 使用保存的索引进行数据分割
            log_message(f"成功加载索引。训练集: {len(train_indices)}个样本, 测试集: {len(test_indices)}个样本", log_file)

            X_train = X_processed[train_indices]
            X_test = X_processed[test_indices]
            y_train = y_values[train_indices]
            y_test = y_values[test_indices]

            log_message(f"使用保存的索引划分数据。训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本",
                        log_file)
        else:
            # 如果无法加载索引，回退到随机分割
            log_message("未找到或无法加载保存的索引，回退到随机分割", log_file)
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_values, test_size=0.2, random_state=42)
            log_message(f"随机分割数据。训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本", log_file)

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 保存缩放器以便将来使用
        joblib.dump(scaler, os.path.join(output_dir, "models", "scaler.pkl"))

        # 保存所有模型评估结果的列表
        all_results = []

        # 1. XGBoost - 使用基础参数
        log_message("\n开始训练 XGBoost 模型...", log_file)

        # 检查GPU可用性
        gpu_available = False
        try:
            # 尝试创建GPU DMatrix
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            param_test = {'max_depth': 2, 'eta': 0.1, 'objective': 'reg:squarederror', 'tree_method': 'gpu_hist'}
            gpu_available = True
            log_message("XGBoost GPU 加速可用，将使用 GPU 训练", log_file)
        except Exception as e:
            log_message(f"XGBoost GPU 加速不可用: {str(e)}", log_file)
            log_message("将使用 CPU 训练 XGBoost", log_file)

        # 基础XGBoost参数
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.5,
            'n_estimators': 100,
            'subsample': 0.2,
            'colsample_bytree': 0.5,
            'tree_method': 'gpu_hist' if gpu_available else 'hist',
            'random_state': 42
        }

        # 创建并训练模型
        xgb_start = time.time()
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_end = time.time()

        log_message(f"XGBoost 训练时间: {xgb_end - xgb_start:.2f} 秒", log_file)

        # 保存模型
        xgb_model.save_model(os.path.join(output_dir, "models", "xgboost_model.json"))

        # 评估模型
        xgb_results, xgb_train_pred, xgb_test_pred = evaluate_model(
            xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost", log_file)
        all_results.append(xgb_results)

        # 2. LightGBM - 使用基础参数
        log_message("\n开始训练 LightGBM 模型...", log_file)

        # 检查GPU可用性
        lgb_gpu_available = False
        try:
            # 尝试创建GPU数据集
            lgb_train = lgb.Dataset(X_train_scaled, y_train)
            lgb_param_test = {'objective': 'regression', 'device': 'gpu'}
            lgb_gpu_available = True
            log_message("LightGBM GPU 加速可用，将使用 GPU 训练", log_file)
        except Exception as e:
            log_message(f"LightGBM GPU 加速不可用: {str(e)}", log_file)
            log_message("将使用 CPU 训练 LightGBM", log_file)

        # 基础LightGBM参数
        lgb_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 3,
            'max_depth': 3,
            'learning_rate': 0.5,
            'n_estimators': 100,
            'subsample': 0.2,
            'colsample_bytree': 0.2,
            'random_state': 42
        }

        # 如果GPU可用，添加GPU参数
        if lgb_gpu_available:
            lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })

        # 创建并训练模型
        lgb_start = time.time()
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train_scaled, y_train)
        lgb_end = time.time()

        log_message(f"LightGBM 训练时间: {lgb_end - lgb_start:.2f} 秒", log_file)

        # 保存模型
        joblib.dump(lgb_model, os.path.join(output_dir, "models", "lightgbm_model.pkl"))

        # 评估模型
        lgb_results, lgb_train_pred, lgb_test_pred = evaluate_model(
            lgb_model, X_train_scaled, y_train, X_test_scaled, y_test, "LightGBM", log_file)
        all_results.append(lgb_results)


        # 创建汇总结果DataFrame
        results_df = pd.DataFrame(all_results)

        # 保存结果到Excel
        results_df.to_excel(os.path.join(output_dir, "model_evaluation_results.xlsx"), index=False)

        # 绘制测试集预测值与真实值的对比图
        plt.figure(figsize=(16, 12))

        # XGBoost
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, xgb_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('XGBoost')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        # LightGBM
        plt.subplot(2, 3, 2)
        plt.scatter(y_test, lgb_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('LightGBM')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        # CatBoost
        plt.subplot(2, 3, 3)
        plt.scatter(y_test, cb_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('CatBoost')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        # Random Forest
        plt.subplot(2, 3, 4)
        plt.scatter(y_test, rf_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Random Forest')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        # Gradient Boosting
        plt.subplot(2, 3, 5)
        plt.scatter(y_test, gb_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Gradient Boosting')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        # AdaBoost
        plt.subplot(2, 3, 6)
        plt.scatter(y_test, ada_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('AdaBoost')
        plt.xlabel('真实值')
        plt.ylabel('预测值')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_vs_true.png"), dpi=300)

        # 绘制模型性能对比图 - 更新显示所有6个模型
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
        try:
            # 使用UTF-8编码写入错误追踪信息
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(traceback.format_exc() + '\n')
        except Exception as e2:
            print(f"无法写入错误日志: {str(e2)}")


if __name__ == "__main__":
    main()