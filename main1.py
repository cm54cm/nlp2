import jieba
import gensim
import os
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ======================
# 数据预处理
# ======================
def load_stopwords(stop_file='stopwords.txt'):
    """加载停用词表"""
    with open(stop_file, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])


def process_corpus(corpus_dir, stopwords):
    """处理语料库，返回分词后的句子列表"""
    sentences = []
    # 添加武侠小说专有名词到分词词典
    jieba.load_userdict('martial_arts_dict.txt')  # 需提前准备

    for file in os.listdir(corpus_dir):
        if file.endswith('.txt'):
            with open(os.path.join(corpus_dir, file), 'r', encoding='gb18030') as f:
                text = f.read()
                # 清洗文本并分句
                text = re.sub(r'\s+', '', text)  # 去除空白字符
                raw_sentences = re.split('[。！？]', text)

                for sent in raw_sentences:
                    if not sent: continue
                    # 分词处理
                    words = [w for w in jieba.lcut(sent)
                             if w not in stopwords
                             and len(w) > 1  # 去除单字
                             and not re.match('[^\u4e00-\u9fa5]', w)]
                    if len(words) > 3:  # 过滤短句
                        sentences.append(words)
    return sentences


# ======================
# 词向量训练
# ======================
def train_word2vec(sentences):
    """训练Word2Vec模型"""
    model = Word2Vec(
        sentences,
        vector_size=200,  # 词向量维度
        window=8,  # 上下文窗口
        min_count=10,  # 最小词频
        workers=8,  # 并行线程数
        epochs=100,  # 训练轮数
        sg=1  # 使用skip-gram
    )
    return model


# ======================
# 验证方法
# ======================
def calculate_similarities(model):
    """计算词语相似度"""
    test_pairs = [
        ('张无忌', '周芷若'),
        ('张无忌', '赵敏'),
        ('灭绝师太', '周芷若'),
        ('灭绝师太', '赵敏'),
        ('张无忌', '明教'),
        ('张无忌', '峨嵋派'),
        ('周芷若', '峨嵋派'),
        ('周芷若', '明教')
    ]

    print("\n词语相似度分析：")
    for w1, w2 in test_pairs:
        try:
            sim = model.wv.similarity(w1, w2)
            print(f"{w1} - {w2}: {sim:.4f}")
        except KeyError as e:
            print(f"缺少词语：{e}")


def visualize_clusters(model):
    """聚类可视化"""
    # 准备测试数据
    categories = {
        '人物': ['张无忌', '赵敏', '周芷若', '谢逊', '殷素素', '张三丰'],
        '武功': ['九阳神功', '乾坤大挪移', '太极拳', '圣火令', '玄冥神掌'],
        '武器': ['倚天剑', '屠龙刀', '圣火令', '铁焰令']
    }

    # 收集词向量
    words, labels, colors = [], [], []
    colors_map = {'人物': 'red', '武功': 'blue', '武器': 'green'}
    for label, (cat, items) in enumerate(colors_map.items()):
        for word in categories[cat]:
            if word in model.wv:
                words.append(word)
                labels.append(label)
                colors.append(colors_map[cat])

    vectors = np.array([model.wv[word] for word in words])

    # 降维可视化
    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    for x, y, word, color in zip(points[:, 0], points[:, 1], words, colors):
        plt.scatter(x, y, c=color, s=80)
        plt.text(x + 0.02, y + 0.02, word, fontsize=10)
    plt.title("词向量聚类可视化")
    plt.show()

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 配置参数
    CORPUS_DIR = '金庸小说'  # 语料库目录
    STOP_FILE = 'stopwords.txt'  # 停用词文件

    # 1. 数据预处理
    stopwords = load_stopwords(STOP_FILE)
    sentences = process_corpus(CORPUS_DIR, stopwords)

    # 2. 训练模型
    print(f"训练语料包含 {len(sentences)} 个句子")
    model = train_word2vec(sentences)
    model.save("martial_arts_word2vec.model")

    # 3. 验证分析
    calculate_similarities(model)
    visualize_clusters(model)


