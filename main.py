import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from zhipuai import ZhipuAI
import random
from dotenv import load_dotenv
import os
import glob

# 加载环境变量
load_dotenv()

# 从环境变量获取API key
client = ZhipuAI(api_key=os.getenv('ZHIPUAI_API_KEY'))

# 获取模型名称
MODEL_NAME = os.getenv('ZHIPUAI_MODEL_NAME', 'glm-4-flash')

# 筛选"相关产品"列的系数
THRESHOLD = float(os.getenv('THRESHOLD', '0.3'))

# 关键词聚类数量
NUM_CLUSTERS = int(os.getenv('NUM_CLUSTERS', '10'))

# 第一部分：清洗表格
file_path = glob.glob('*.xlsx')[0]  # 获取当前目录第一个xlsx文件
xls = pd.ExcelFile(file_path)

# 保留第一个sheet, 读取数据时不使用默认标题行
df = pd.read_excel(xls, sheet_name=0, header=None)

# 删除第一行
df = df.iloc[1:].reset_index(drop=True)

# 重新设置列名
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# 填充空值为0
df = df.fillna(0)

# 移除"PPC竞价"列中的货币符号
df['PPC竞价'] = df['PPC竞价'].replace('[\$,]', '', regex=True).astype(float)

# 筛选"相关产品"列的值为最大数值的30%
threshold = df['相关产品'].max() * THRESHOLD
df_filtered = df[df['相关产品'] >= threshold]

# 按"流量占比"降序排列
df_sorted = df_filtered.sort_values(by='流量占比', ascending=False)

# 保留需要的列
columns_to_keep = [
    '关键词', '关键词翻译', '流量占比', '预估周曝光量', 'ABA周排名', 
    '月搜索量', '月购买量', '购买率', '展示量', '点击量', 'SPR', 
    '商品数', '需供比', '广告竞品数', 
    '点击总占比', '转化总占比', 'PPC竞价'
]
df = df_sorted[columns_to_keep].copy()  # 使用.copy()创建一个新的DataFrame

# 第二部分：聚类和大模型打标签

# 下载必要的NLTK数据
# nltk.download('punkt', quiet=True)

# 设置 NLTK 数据目录为当前项目下的 nltk 文件夹
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk'))

# 创建词干提取器
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    return ' '.join([ps.stem(word) for word in words])

def select_representative_keywords(keywords, max_keywords=100, max_chars=1000):
    if len(keywords) <= max_keywords:
        selected = keywords
    else:
        selected = random.sample(keywords, max_keywords)
    
    total_chars = sum(len(kw) for kw in selected)
    while total_chars > max_chars and len(selected) > 20:
        selected.pop()
        total_chars = sum(len(kw) for kw in selected)
    
    return selected

# 使用智谱AI为聚类添加标签
def get_cluster_label(keywords):
    selected_keywords = select_representative_keywords(keywords)
    prompt = f"""
    以下是一组相关的关键词（共{len(keywords)}个，显示{len(selected_keywords)}个样本）：
    {', '.join(selected_keywords)}

    请根据这些关键词的共同特征，为它们创建一个简洁而准确的分类标签。
    这个标签应该：
    1. 反映这组关键词的核心主题或共同点
    2. 不超过5个字
    3. 尽可能具体和有描述性
    4. 可以是产品类别、功能特征、使用场景、用户需求等任何相关的分类

    请直接给出分类标签，无需其他解释。
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        
        label = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                label += chunk.choices[0].delta.content
        
        return label.strip()
    except Exception as e:
        return "未分类"
    
# 预处理关键词
keywords = df['关键词'].apply(preprocess_text)

# 创建TF-IDF向量
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(keywords)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(X)

# 将聚类结果添加到DataFrame中
df['原始聚类编号'] = kmeans.labels_

# 为每个聚类添加标签
cluster_labels = {}
for i in range(NUM_CLUSTERS):
    cluster_keywords = df[df['原始聚类编号'] == i]['关键词'].tolist()
    label = get_cluster_label(cluster_keywords)
    cluster_labels[i] = label

# 将分类标签添加到DataFrame中
df['关键词分类'] = df['原始聚类编号'].map(cluster_labels)

# 计算每个聚类的流量占比总和
cluster_traffic = df.groupby('原始聚类编号')['流量占比'].sum().sort_values(ascending=False)

# 创建新的聚类编号映射
new_cluster_numbers = {old: new+1 for new, old in enumerate(cluster_traffic.index)}

# 更新DataFrame中的聚类编号
df['聚类编号'] = df['原始聚类编号'].map(new_cluster_numbers)

# 更新分类标签的顺序
ordered_labels = {new_cluster_numbers[old]: label for old, label in cluster_labels.items()}

# 重新排序列
df = df[['关键词', '聚类编号', '关键词分类', '流量占比'] + [col for col in df.columns if col not in ['关键词', '聚类编号', '关键词分类', '流量占比', '原始聚类编号']]]

# 保存结果到新的CSV文件
df.to_csv('output.csv', index=False)

print("表格处理完成，结果已保存")
