# ===================== 1. 导入依赖包 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA as SKPCA
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== 2. 基础配置 =====================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（对接PCA/因子分析结果）
PCA_SCORES_PATH = "D:/PythonCode/demo/results/chapter6/PCA主成分得分矩阵.csv"
INPUT_DATA_PATH = "D:/PythonCode/demo/data/processed/共线性处理后最终数据.csv"
RESULT_DIR = "D:/PythonCode/demo/results/chapter7/"
os.makedirs(RESULT_DIR, exist_ok=True)

# 固定随机种子
np.random.seed(42)

# ===================== 3. 核心函数 =====================
def load_data():
    """加载PCA得分和原始数据，合并标识列（修复：确保df_pca仅含数值）"""
    # 加载PCA得分（仅保留数值列）
    df_pca = pd.read_csv(PCA_SCORES_PATH, encoding='utf-8-sig', index_col=0)
    # 过滤非数值列（防止PCA得分文件混入字符串）
    df_pca = df_pca.select_dtypes(include=[np.float64, np.int64])
    
    # 加载原始数据（获取标识列）
    df_raw = pd.read_csv(INPUT_DATA_PATH, encoding='utf-8-sig')
    id_cols = [col for col in df_raw.columns if "城市" in col or "年份" in col]
    df_id = df_raw[id_cols].reset_index(drop=True)
    
    # 合并数据（标识列+PCA得分）
    df = pd.concat([df_id, df_pca.reset_index(drop=True)], axis=1)
    return df, df_pca  # df_pca仅含数值型PCA得分

def optimal_kmeans_clusters(df_numeric, max_k=10):
    """自动选择最优K值（肘部法则+轮廓系数）"""
    inertia = []  # 惯性值（越小越好）
    silhouette = []  # 轮廓系数（越接近1越好）
    k_range = range(2, max_k+1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_numeric)  # 仅传入数值型数据
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(df_numeric, labels))
    
    # 可视化肘部法则
    plt.figure(figsize=(12, 5))
    # 子图1：肘部法则
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('聚类数K')
    plt.ylabel('惯性值')
    plt.title('KMeans肘部法则（最优K选择）')
    plt.xticks(k_range)
    
    # 子图2：轮廓系数
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette, 'o-', color='r')
    plt.xlabel('聚类数K')
    plt.ylabel('轮廓系数')
    plt.title('KMeans轮廓系数（最优K选择）')
    plt.xticks(k_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "最优K值选择.png"), dpi=300)
    plt.close()
    
    # 选择最优K（轮廓系数最大值对应的K）
    optimal_k = k_range[np.argmax(silhouette)]
    print(f"最优聚类数K：{optimal_k}（轮廓系数={max(silhouette):.4f}）")
    return optimal_k

def kmeans_clustering(df_pca, k):
    """KMeans聚类分析"""
    # 数据标准化（聚类必须）
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)
    
    # 执行KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=df_pca.columns,
        index=[f"聚类{i+1}中心" for i in range(k)]
    )
    
    # 可视化聚类结果（降维到2D）
    pca_2d = SKPCA(n_components=2, random_state=42).fit_transform(df_scaled)
    plt.figure(figsize=(10, 8))
    for i in range(k):
        plt.scatter(pca_2d[cluster_labels==i, 0], pca_2d[cluster_labels==i, 1], 
                    label=f"聚类{i+1}", alpha=0.7, s=60)
    # 绘制聚类中心
    centers_2d = SKPCA(n_components=2, random_state=42).fit_transform(kmeans.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                marker='*', s=200, c='black', label='聚类中心')
    plt.xlabel('PCA降维维度1')
    plt.ylabel('PCA降维维度2')
    plt.title(f'KMeans聚类结果（K={k}）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "KMeans聚类结果.png"), dpi=300)
    plt.close()
    
    # 统计各聚类样本数
    cluster_count = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_count.index = [f"聚类{i+1}" for i in cluster_count.index]
    print("\n各聚类样本数：")
    print(cluster_count)
    
    return cluster_labels, cluster_centers, cluster_count

def classification_analysis(df_pca, cluster_labels):
    """基于聚类标签的逻辑回归分类"""
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df_pca, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
    )
    
    # 逻辑回归分类
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # 模型评估
    print("\n===== 分类模型评估 =====")
    print("分类报告：")
    print(classification_report(y_test, y_pred, target_names=[f"聚类{i+1}" for i in np.unique(cluster_labels)]))
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"聚类{i+1}" for i in np.unique(cluster_labels)],
                yticklabels=[f"聚类{i+1}" for i in np.unique(cluster_labels)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('逻辑回归分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "分类混淆矩阵.png"), dpi=300)
    plt.close()
    
    # 特征重要性
    feature_importance = pd.DataFrame(
        lr.coef_.mean(axis=0),  # 多分类取平均系数
        index=df_pca.columns,
        columns=['特征重要性']
    ).sort_values('特征重要性', ascending=False)
    
    return feature_importance, lr.score(X_test, y_test)

def save_results(df, cluster_labels, cluster_centers, cluster_count, feature_importance, test_score):
    """保存所有结果"""
    # 合并聚类标签
    df['聚类标签'] = cluster_labels
    df.to_csv(os.path.join(RESULT_DIR, "聚类结果完整数据.csv"), encoding='utf-8-sig', index=False)
    
    # 聚类中心
    cluster_centers.to_csv(os.path.join(RESULT_DIR, "聚类中心.csv"), encoding='utf-8-sig')
    
    # 聚类样本数
    cluster_count.to_csv(os.path.join(RESULT_DIR, "各聚类样本数.csv"), encoding='utf-8-sig', header=['样本数'])
    
    # 特征重要性
    feature_importance.to_csv(os.path.join(RESULT_DIR, "分类特征重要性.csv"), encoding='utf-8-sig')
    
    # 分类模型得分
    with open(os.path.join(RESULT_DIR, "分类模型评估.txt"), "w", encoding='utf-8') as f:
        f.write(f"分类模型测试集准确率：{test_score:.4f}\n")
        f.write(f"最优聚类数K：{len(cluster_count)}\n")
        f.write(f"各聚类样本数：\n{cluster_count.to_string()}")

# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    df, df_pca = load_data()
    print(f"聚类输入数据：{df_pca.shape[0]}行×{df_pca.shape[1]}列")
    
    # 2. 选择最优聚类数
    optimal_k = optimal_kmeans_clusters(df_numeric=df_pca)
    
    # 3. KMeans聚类
    cluster_labels, cluster_centers, cluster_count = kmeans_clustering(df_pca, optimal_k)
    
    # 4. 分类分析（逻辑回归）
    feature_importance, test_score = classification_analysis(df_pca, cluster_labels)
    
    # 5. 保存结果
    save_results(df, cluster_labels, cluster_centers, cluster_count, feature_importance, test_score)
    
    # 6. 输出总结
    print("\n✅ 聚类+分类分析完成！结果已保存")
    print(f"核心结论：最优聚类数K={optimal_k}，分类模型准确率={test_score:.4f}")