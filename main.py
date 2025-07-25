import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import random
from dotenv import load_dotenv
import os
import glob


def load_config():
    """加载环境变量配置"""
    load_dotenv()
    
    config = {
        'API_KEY': os.getenv('API_KEY', ''),
        'MODEL': os.getenv('MODEL', 'gpt-3.5-turbo'),
        'BASE_URL': os.getenv('BASE_URL', 'https://api.openai.com/v1'),
        'THRESHOLD': float(os.getenv('THRESHOLD', '0.3')),
        'NUM_CLUSTERS': int(os.getenv('NUM_CLUSTERS', '10'))
    }
    
    return config


def create_llm_client(config):
    """创建兼容OpenAI接口的客户端"""
    from openai import OpenAI
    
    if not config['BASE_URL']:
        raise ValueError("必须提供BASE_URL")
        
    client = OpenAI(
        api_key=config['API_KEY'],
        base_url=config['BASE_URL']
    )
    
    print(f"成功连接到模型服务，端点: {config['MODEL']}")
    return client


def setup_nltk():
    """设置NLTK环境和数据"""
    # 设置 NLTK 数据目录为当前工作目录下的 nltk 文件夹
    nltk_data_path = os.path.join(os.getcwd(), 'nltk')
    
    # 确保nltk目录存在
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    # 检查是否已下载punkt分词器
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载必要的NLTK数据...")
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        print("下载完成")


def load_and_clean_data(threshold):
    """加载并清洗数据"""
    # 获取当前目录第一个xlsx文件
    xlsx_files = glob.glob('*.xlsx')
    if not xlsx_files:
        raise FileNotFoundError("当前目录没有找到xlsx文件")
    
    file_path = xlsx_files[0]
    print(f"正在处理文件: {file_path}")
    
    xls = pd.ExcelFile(file_path)
    
    # 保留第一个sheet, 读取数据时不使用默认标题行
    df = pd.read_excel(xls, sheet_name=0, header=None)
    
    # 删除第一行
    df = df.iloc[1:].reset_index(drop=True)
    
    # 重新设置列名
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    
    # 填充空值为0 (避免FutureWarning)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
                
    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    
    # 移除"PPC竞价"列中的货币符号（支持多种货币）
    # 支持的货币符号：A$（澳洲）、R$（巴西）、MX$（墨西哥）、CDN$（加拿大）、€（欧元）、￥（日元）、£（英镑）、$（美元）
    df['PPC竞价'] = df['PPC竞价'].replace(r'A\$|R\$|MX\$|CDN\$|€|￥|£|\$|,', '', regex=True).astype(float)
    
    # 筛选"相关产品"列的值为最大数值的指定百分比
    threshold_value = df['相关产品'].max() * threshold
    df_filtered = df[df['相关产品'] >= threshold_value]
    print(f"筛选后保留了 {len(df_filtered)} 个关键词")
    
    # 按"流量占比"降序排列
    df_sorted = df_filtered.sort_values(by='流量占比', ascending=False)
    
    # 保留需要的列
    columns_to_keep = [
        '关键词', '关键词翻译', '流量占比', '预估周曝光量', 'ABA周排名', 
        '月搜索量', '月购买量', '购买率', '展示量', '点击量', 'SPR', 
        '商品数', '需供比', '广告竞品数', 
        '点击总占比', '转化总占比', 'PPC竞价'
    ]
    df_clean = df_sorted[columns_to_keep].copy()
    
    return df_clean


def preprocess_text(text):
    """预处理文本：分词和词干提取"""
    ps = PorterStemmer()
    words = word_tokenize(text.lower())
    return ' '.join([ps.stem(word) for word in words])


def select_representative_keywords(keywords, max_keywords=100, max_chars=1000):
    """选择代表性关键词用于标签生成"""
    if len(keywords) <= max_keywords:
        selected = keywords
    else:
        selected = random.sample(keywords, max_keywords)
    
    total_chars = sum(len(kw) for kw in selected)
    while total_chars > max_chars and len(selected) > 20:
        selected.pop()
        total_chars = sum(len(kw) for kw in selected)
    
    return selected


def get_cluster_label(keywords, client, model):
    """使用大模型API为聚类生成标签"""
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
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"调用模型API出错: {e}")
        return "未分类"


def perform_clustering_and_labeling(df, client, config):
    """执行聚类和标签生成"""
    print("开始聚类分析...")
    
    # 预处理关键词
    keywords = df['关键词'].apply(preprocess_text)
    
    # 创建TF-IDF向量
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keywords)
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=config['NUM_CLUSTERS'], random_state=42)
    kmeans.fit(X)
    
    # 将聚类结果添加到DataFrame中
    df['原始聚类编号'] = kmeans.labels_
    
    # 为每个聚类添加标签
    cluster_labels = {}
    for i in range(config['NUM_CLUSTERS']):
        cluster_keywords = df[df['原始聚类编号'] == i]['关键词'].tolist()
        label = get_cluster_label(cluster_keywords, client, config['MODEL'])
        cluster_labels[i] = label
        print(f"完成聚类 {i+1}/{config['NUM_CLUSTERS']} 的标签生成: {label}")
    
    # 将分类标签添加到DataFrame中
    df['关键词分类'] = df['原始聚类编号'].map(cluster_labels)
    
    # 计算每个聚类的流量占比总和，按流量占比重新排序聚类编号
    cluster_traffic = df.groupby('原始聚类编号')['流量占比'].sum().sort_values(ascending=False)
    
    # 创建新的聚类编号映射
    new_cluster_numbers = {old: new+1 for new, old in enumerate(cluster_traffic.index)}
    
    # 更新DataFrame中的聚类编号
    df['聚类编号'] = df['原始聚类编号'].map(new_cluster_numbers)
    
    # 重新排序列
    df = df[['关键词', '聚类编号', '关键词分类', '流量占比'] + 
           [col for col in df.columns if col not in ['关键词', '聚类编号', '关键词分类', '流量占比', '原始聚类编号']]]
    
    return df


def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        print("配置加载完成")
        
        # 创建LLM客户端
        client = create_llm_client(config)
        
        # 设置NLTK环境
        setup_nltk()
        
        # 加载和清洗数据
        df = load_and_clean_data(config['THRESHOLD'])
        
        # 执行聚类和标签生成
        df_final = perform_clustering_and_labeling(df, client, config)
        
        # 保存结果
        output_file = 'output.csv'
        df_final.to_csv(output_file, index=False)
        
        print(f"处理完成！结果已保存到 {output_file}")
        print(f"共处理了 {len(df_final)} 个关键词，分为 {config['NUM_CLUSTERS']} 个聚类")
        
        # 显示聚类分布
        cluster_summary = df_final.groupby(['聚类编号', '关键词分类']).agg({
            '关键词': 'count',
            '流量占比': 'sum'
        }).round(2)
        print("\n聚类分布:")
        print(cluster_summary)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 