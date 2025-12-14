# ===================== 1. 导入依赖包 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== 2. 基础配置 =====================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（对接预处理结果）
INPUT_DATA_PATH = "D:/PythonCode/demo/data/processed/共线性处理后最终数据.csv"
RESULT_DIR = "D:/PythonCode/demo/results/chapter6/"
os.makedirs(RESULT_DIR, exist_ok=True)

# 固定随机种子
np.random.seed(42)

# ===================== 3. 核心函数 =====================
def load_preprocessed_data(file_path):
    """加载预处理后的数据，分离标识列和数值列"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    # 分离标识列（城市/年份）和数值列
    id_cols = [col for col in df.columns if "城市" in col or "年份" in col]
    num_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col not in id_cols]
    return df[id_cols], df[num_cols]

def save_result(df, file_name):
    """保存结果到指定目录"""
    df.to_csv(os.path.join(RESULT_DIR, file_name), encoding='utf-8-sig', index=True)

def pca_analysis(df_num, n_components=None):
    """PCA主成分分析"""
    # 1. 数据标准化（PCA必须）
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)
    
    # 2. 执行PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(df_scaled)
    
    # 3. 提取核心指标
    # 主成分方差贡献率
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var_ratio)
    # 主成分载荷矩阵
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"主成分{i+1}" for i in range(len(exp_var_ratio))],
        index=df_num.columns
    )
    
    # 4. 输出关键信息
    print("===== PCA分析结果 =====")
    print(f"累计方差贡献率：{cum_exp_var}")
    print(f"保留{n_components if n_components else len(exp_var_ratio)}个主成分，累计解释方差：{cum_exp_var[-1]:.4f}")
    
    # 5. 可视化方差贡献率
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(exp_var_ratio)+1), exp_var_ratio, alpha=0.7, label='单个方差贡献率')
    plt.plot(range(1, len(exp_var_ratio)+1), cum_exp_var, 'r-', marker='o', label='累计方差贡献率')
    plt.axhline(y=0.8, color='g', linestyle='--', label='80%阈值')
    plt.xlabel('主成分个数')
    plt.ylabel('方差贡献率')
    plt.title('PCA主成分方差贡献率分布')
    plt.legend()
    plt.xticks(range(1, len(exp_var_ratio)+1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "PCA方差贡献率.png"), dpi=300)
    plt.close()
    
    # 6. 构造主成分得分矩阵（合并标识列）
    pca_scores = pd.DataFrame(pca_result, columns=[f"主成分{i+1}得分" for i in range(len(exp_var_ratio))])
    
    return loadings, pca_scores, cum_exp_var[-1]

def factor_analysis(df_num, n_factors=None):
    """因子分析（探索性）- 兼容新旧版本factor_analyzer"""
    # 1. 数据标准化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)
    
    # 2. 适用性检验（Bartlett球形检验+KMO检验）
    bartlett_stat, bartlett_p = calculate_bartlett_sphericity(df_scaled)
    kmo_all, kmo_model = calculate_kmo(df_scaled)
    if isinstance(kmo_model, np.ndarray):
        kmo_val = np.mean(kmo_model)
    else:
        kmo_val = kmo_model
    
    print("\n===== 因子分析适用性检验 =====")
    print(f"Bartlett球形检验：统计量={bartlett_stat:.4f}, p值={bartlett_p:.4f}（p<0.05说明适合因子分析）")
    print(f"KMO检验值：{kmo_val:.4f}（>0.6说明适合因子分析）")
    
    if kmo_val < 0.6 or bartlett_p >= 0.05:
        print("警告：数据不适合做因子分析！")
        return None, None, None  # 保证返回值数量匹配
    
    # 3. 执行因子分析（最大方差旋转）
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(df_scaled)
    
    # 4. 提取核心指标
    # 公因子方差（共同度）
    communality = pd.DataFrame(
        fa.get_communalities(),
        index=df_num.columns,
        columns=['公因子方差']
    )
    # 因子载荷矩阵
    loadings = pd.DataFrame(
        fa.loadings_,
        index=df_num.columns,
        columns=[f"公因子{i+1}" for i in range(n_factors if n_factors else fa.n_factors)]
    )
    # 因子得分
    fa_scores = pd.DataFrame(
        fa.transform(df_scaled),
        columns=[f"公因子{i+1}得分" for i in range(n_factors if n_factors else fa.n_factors)]
    )
    
    # 5. 可视化因子载荷矩阵（热力图）
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('因子载荷矩阵（最大方差旋转）')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "因子载荷矩阵.png"), dpi=300)
    plt.close()
    
    print("\n===== 因子分析结果 =====")
    print(f"提取{n_factors if n_factors else fa.n_factors}个公因子，平均公因子方差：{communality['公因子方差'].mean():.4f}")
    
    return communality, loadings, fa_scores

# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    df_id, df_num = load_preprocessed_data(INPUT_DATA_PATH)
    print(f"因子分析输入数据：{df_num.shape[0]}行×{df_num.shape[1]}列")
    
    # 2. PCA分析（自动选择保留80%方差的主成分）
    # 先跑一遍PCA确定最优主成分数
    temp_pca = PCA(random_state=42).fit(StandardScaler().fit_transform(df_num))
    cum_var = np.cumsum(temp_pca.explained_variance_ratio_)
    n_pca = np.argmax(cum_var >= 0.8) + 1  # 找到累计方差≥80%的最小主成分数
    
    pca_loadings, pca_scores, pca_cum_var = pca_analysis(df_num, n_components=n_pca)
    # 保存PCA结果
    save_result(pca_loadings, "PCA主成分载荷矩阵.csv")
    save_result(pd.concat([df_id.reset_index(drop=True), pca_scores], axis=1), "PCA主成分得分矩阵.csv")
    
    # 3. 因子分析（与PCA保留相同个数的公因子）
    fa_communalities, fa_loadings, fa_scores = factor_analysis(df_num, n_factors=n_pca)
    if fa_loadings is not None:
        # 保存因子分析结果
        save_result(fa_communalities, "因子分析公因子方差.csv")
        save_result(fa_loadings, "因子分析载荷矩阵.csv")
        save_result(pd.concat([df_id.reset_index(drop=True), fa_scores], axis=1), "因子分析得分矩阵.csv")
    
    # 4. 输出总结
    print("\n✅ PCA+因子分析完成！结果已保存")
    print(f"核心结论：保留{n_pca}个主成分/公因子，PCA累计解释方差{pca_cum_var:.4f}")