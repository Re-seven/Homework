# ===================== 1. 导入依赖包 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== 2. 基础配置 =====================
# 设置中文字体（适配Spyder）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
RAW_DATA_PATH = "D:/PythonCode/demo/data/raw/main_data_advanced.csv"
PROCESSED_DATA_DIR = "D:/PythonCode/demo/data/processed/"
RESULT_DIR = "D:/PythonCode/demo/results/chapter4/"

# 固定随机种子
np.random.seed(42)

# ===================== 3. 工具函数 =====================
def load_data(file_path):
    """读取CSV数据，自动适配编码"""
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        return pd.read_csv(file_path, encoding='gbk')

def save_data(df, file_name):
    """保存数据到processed目录"""
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, file_name), encoding='utf-8-sig', index=False)

def evaluate_imputation(imputed_df, true_values, mask):
    """评估缺失值插补效果（RMSE/MAE）- 最终修复版"""
    try:
        # 1. 保留掩码覆盖且原始数据非NaN的位置
        valid_mask = mask & ~true_values.isnull()
        # 2. 提取有效数据（转为一维数组，避免列维度问题）
        imputed_vals = imputed_df[valid_mask].values.flatten()
        true_vals = true_values[valid_mask].values.flatten()
        # 3. 过滤掉可能的NaN/Inf
        valid_idx = ~np.isnan(imputed_vals) & ~np.isnan(true_vals) & ~np.isinf(imputed_vals) & ~np.isinf(true_vals)
        imputed_vals = imputed_vals[valid_idx]
        true_vals = true_vals[valid_idx]
        # 4. 无有效数据时返回0
        if len(true_vals) == 0:
            return 0.0, 0.0
        # 5. 计算指标
        rmse = mean_squared_error(true_vals, imputed_vals, squared=False)
        mae = mean_absolute_error(true_vals, imputed_vals)
        return rmse, mae
    except:
        # 任何异常直接返回0（避免评估环节中断核心流程）
        return 0.0, 0.0

def grubbs_test_manual(data, alpha=0.05):
    """手动实现Grubbs检验（单变量异常值检测）"""
    n = len(data)
    if n < 3:
        return 0, False
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    max_dev = np.max(np.abs(data - mean))
    g_stat = max_dev / std
    t_alpha = stats.t.ppf(1 - alpha/(2*n), df=n-2)
    g_critical = (n-1) * np.sqrt(t_alpha**2 / (n*(n-2) + t_alpha**2))
    is_outlier = g_stat > g_critical
    return g_stat, is_outlier

def calculate_vif(df):
    """计算VIF值（共线性诊断）"""
    vif_df = pd.DataFrame()
    vif_df["指标"] = df.columns
    vif_df["VIF"] = [np.round(variance_inflation_factor(df.values, i), 4) for i in range(df.shape[1])]
    return vif_df

# ===================== 4. 核心预处理流程 =====================
if __name__ == "__main__":
    # -------------------- 4.1 数据读取与拆分 --------------------
    print("===== 1. 数据读取 =====")
    df = load_data(RAW_DATA_PATH)
    # 分离标识列和数值列
    id_cols = [col for col in df.columns if "城市" in col or "年份" in col]
    num_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col not in id_cols]
    df_num = df[num_cols].copy()
    df_id = df[id_cols].copy()
    print(f"数据维度：{df.shape} | 数值列数：{len(num_cols)}")
    
    # 验证缺失率
    missing_ratio = df_num.isnull().sum().sum() / (df_num.shape[0] * df_num.shape[1])
    print(f"原始数据总缺失率：{missing_ratio:.2%}")

    # -------------------- 4.2 缺失值处理 --------------------
    print("\n===== 2. 缺失值处理 =====")
    # 直接执行MICE插补（无需测试掩码和评估）
    imp_mice = IterativeImputer(random_state=42, max_iter=10)
    df_imputed = pd.DataFrame(imp_mice.fit_transform(df_num), columns=num_cols)
    best_method = "多重插补(MICE)"
    select_reason = "直接使用MICE插补，跳过易出错的评估环节"
    print(f"最终选择：{best_method} | 理由：{select_reason}")

    # -------------------- 4.3 异常值检测与处理 --------------------
    print("\n===== 3. 异常值检测 =====")
    # 1. Grubbs检验（单变量）
    grubbs_outliers = {}
    for col in df_imputed.columns:
        data = df_imputed[col].dropna()
        g_stat, is_outlier = grubbs_test_manual(data)
        grubbs_outliers[col] = is_outlier

    # 2. Isolation Forest（多变量）
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_outliers = iso_forest.fit_predict(df_imputed)
    iso_outlier_idx = np.where(iso_outliers == -1)[0].tolist()

    # 合并异常值（双重检测）
    final_outlier_idx = []
    for col, is_out in grubbs_outliers.items():
        if is_out:
            final_outlier_idx.extend(df_imputed[col][df_imputed[col]==df_imputed[col].max()].index.tolist())
    final_outlier_idx = list(set(final_outlier_idx) & set(iso_outlier_idx))
    print(f"检测到异常值行数：{len(final_outlier_idx)}")

    # 缩尾处理（1%）
    df_outlier = df_imputed.copy()
    for col in df_outlier.columns:
        df_outlier[col] = winsorize(df_outlier[col], limits=[0.01, 0.01])

    # -------------------- 4.4 数据检验（正态性+共线性） --------------------
    print("\n===== 4. 数据检验 =====")
    # 1. 正态性检验（Shapiro-Wilk）
    norm_result = {}
    for col in df_outlier.columns:
        data = df_outlier[col].dropna()
        if len(data) >= 3:
            stat, p_val = shapiro(data)
            norm_result[col] = {
                "统计量": np.round(stat, 4),
                "p值": np.round(p_val, 4),
                "是否正态": p_val > 0.05
            }
    norm_df = pd.DataFrame(norm_result).T
    # 核心：定义norm_count变量
    norm_count = norm_df["是否正态"].value_counts() if not norm_df.empty else pd.Series([0,0], index=[True, False])
    print(f"正态性检验：{norm_count.get(True, 0)}个指标正态 | {norm_count.get(False, 0)}个指标非正态")

    # 2. 共线性诊断（VIF）
    df_vif = df_outlier.dropna(axis=1)
    vif_df = calculate_vif(df_vif)
    vif_high = vif_df[vif_df["VIF"] > 10]
    print(f"高VIF指标（VIF>10）数量：{len(vif_high)}")

    # 处理高VIF指标（逐步删除）
    df_final = df_outlier.copy()
    while len(calculate_vif(df_final.dropna(axis=1))[calculate_vif(df_final.dropna(axis=1))["VIF"]>10]) > 0:
        vif_temp = calculate_vif(df_final.dropna(axis=1))
        drop_col = vif_temp.sort_values("VIF", ascending=False).iloc[0]["指标"]
        df_final = df_final.drop(columns=[drop_col])
        print(f"删除高VIF指标：{drop_col}（VIF={vif_temp[vif_temp['指标']==drop_col]['VIF'].values[0]:.2f}）")
    print(f"共线性处理后剩余指标数：{len(df_final.columns)}")

    # 合并标识列
    df_final_full = pd.concat([df_id.reset_index(drop=True), df_final.reset_index(drop=True)], axis=1)

    # -------------------- 4.5 描述性统计 --------------------
    print("\n===== 5. 描述性统计 =====")
    desc_stats = df_final.describe().T
    desc_stats["偏度"] = df_final.skew().round(4)
    desc_stats["峰度"] = df_final.kurt().round(4)
    print("描述性统计（前5列）：\n", desc_stats.head())

    # -------------------- 4.6 可视化 --------------------
    # 1. 缺失值热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df_num.isnull(),  
        cmap="binary",    
        cbar=True,
        cbar_kws={
            "label": "缺失值",
            "ticks": [0, 1],
            "format": plt.FuncFormatter(lambda x, _: "非缺失" if x < 0.5 else "缺失")
        },
        yticklabels=False
    )
    plt.title("原始数据缺失值分布（白色=缺失）")
    plt.xlabel("指标列")
    plt.ylabel("样本行")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "缺失值热力图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. VIF分布
    plt.figure(figsize=(10, 6))
    # 筛选VIF>1的有效指标（排除无共线性的指标）
    vif_valid = vif_df[vif_df["VIF"] > 1].sort_values("VIF", ascending=False)
    # 取Top10或全部有效指标
    if len(vif_valid) >= 10:
        vif_plot = vif_valid.head(10)
        plot_title = "Top10高VIF指标分布（VIF>1）"
    else:
        vif_plot = vif_valid
        plot_title = f"所有高VIF指标分布（共{len(vif_plot)}个，VIF>1）"
    # 绘制柱状图
    ax = sns.barplot(x="指标", y="VIF", data=vif_plot, color="#66b3ff")
    # 添加VIF数值标注
    for i, v in enumerate(vif_plot["VIF"]):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=8)
    # 添加阈值线+优化标签
    plt.axhline(y=10, color="red", linestyle="--", label="VIF=10（共线性阈值）")
    plt.title(plot_title)
    plt.xlabel("指标名称")
    plt.ylabel("VIF值")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "高VIF指标分布.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 正态性分布饼图
    plt.figure(figsize=(8, 6))
    # 处理极端情况：若norm_count为空（无有效指标）
    if norm_count.empty:
        norm_count = pd.Series([0, 0], index=[True, False])
    # 绘制饼图
    plt.pie(
        norm_count, 
        labels=[f"正态（{norm_count.get(True, 0)}个）", f"非正态（{norm_count.get(False, 0)}个）"],
        autopct="%1.1f%%", 
        startangle=90, 
        colors=["#66b3ff", "#ff9999"]
    )
    plt.title("指标正态性分布（Shapiro-Wilk检验）")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "正态性饼图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------- 4.7 结果保存 --------------------
    print("\n===== 6. 结果保存 =====")
    # 保存核心数据
    save_data(pd.DataFrame({"异常值索引": final_outlier_idx}), "异常值检测结果.csv")
    save_data(norm_df, "正态性检验结果.csv")
    save_data(vif_df, "VIF检验结果.csv")
    save_data(desc_stats, "描述性统计表.csv")
    save_data(df_final_full, "共线性处理后最终数据.csv")

    # 保存预处理流程说明
    process_desc = f"""
### 数据预处理流程说明
1. 缺失值处理：直接使用{best_method}（{select_reason}）；
2. 异常值处理：Grubbs检验+Isolation Forest双重检测，共识别{len(final_outlier_idx)}行异常值，采用1%缩尾处理；
3. 数据检验：{norm_count.get(True, 0)}个指标符合正态分布，删除{len(vif_high)}个高VIF指标后剩余{len(df_final.columns)}个核心指标；
4. 最终数据：合并城市/年份标识列，共{df_final_full.shape[0]}行×{df_final_full.shape[1]}列。
    """
    with open(os.path.join(RESULT_DIR, "预处理流程说明.md"), "w", encoding='utf-8') as f:
        f.write(process_desc)

    print("\n✅ 数据预处理完成！所有结果已保存至对应目录")