# 亚马逊关键词库清洗工具 (MIT License)

这是一个基于机器学习的亚马逊关键词聚类分析工具，可以帮助卖家更好地理解和管理产品关键词库。该工具通过分析关键词数据，自动对关键词进行分类，帮助卖家发现高价值的关键词机会。

## 功能特点

- **数据清洗**：自动处理和清洗Excel格式的关键词数据
- **智能聚类**：使用K-means算法对关键词进行智能分组
- **自动标签**：使用智谱AI为每个关键词组自动生成描述性标签
- **流量分析**：基于流量占比进行关键词排序和筛选

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
- 将数据文件重命名为 `data.xlsx` 并放置在项目根目录

### 3. 替换API密钥
- 请修改 `.env` 文件中的 `ZHIPUAI_API_KEY`

### 4. 运行程序
- 点击界面顶部的 "Run" 按钮
- 或在终端中执行：
```bash
python main.py
```

## 输出结果
程序运行完成后会在根目录生成 `output.csv` 文件，包含：
- 关键词聚类结果
- 智能分类标签
- 流量分析数据
- 其他相关指标

## 更多资源

- 🌐 博客：[跨境Ai视界](https://www.amzalysis.com/)
- 📱 公众号：跨境Ai视界
- 📖 详细教程：[亚马逊广告关键词库SEO终极指南](https://www.amzalysis.com/article/amazon-keyword-library)

## License
MIT License - 详见 [LICENSE](LICENSE) 文件