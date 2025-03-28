import os
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from gensim import corpora, models
from joblib import Parallel, delayed


def load_corpus(corpus_dir="./jyxstxtqj_downcc.com/"):
    """加载金庸小说语料库"""
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt'):
            novel_name = file_name[:-4]
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='gb18030') as f:
                text = f.read().replace('\n', '').replace(' ', '')
            novels.append((novel_name, text))
    return novels


def preprocess_text(text, unit="word"):
    """对文本进行分词或以字为单位切分"""
    if unit == "word":
        return list(jieba.cut(text))
    elif unit == "char":
        return list(text)


def sample_paragraphs_single_novel(novel, K=100, num_samples=1000, unit="word"):
    novel_name, text = novel
    tokens = preprocess_text(text, unit)
    num_paragraphs = len(tokens) // K
    sampled = [tokens[i * K:(i + 1) * K] for i in range(num_paragraphs)]
    sampled = random.sample(sampled, min(len(sampled), num_samples // len(novels)))
    return sampled, [novel_name] * len(sampled)


def sample_paragraphs(novels, K=100, num_samples=1000, unit="word"):
    """均匀抽取段落，并行处理"""
    results = Parallel(n_jobs=-1)(delayed(sample_paragraphs_single_novel)(
        novel, K, num_samples, unit) for novel in novels)
    paragraphs = []
    labels = []
    for para, label in results:
        paragraphs.extend(para)
        labels.extend(label)
    return paragraphs, labels


def train_lda(paragraphs, num_topics):
    """训练 LDA 模型"""
    dictionary = corpora.Dictionary(paragraphs)
    corpus = [dictionary.doc2bow(para) for para in paragraphs]
    # 调整 LDA 模型的训练参数，减少训练时间
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                                passes=20,  # 减少迭代次数
                                alpha='auto',
                                eta='auto')
    return lda_model, dictionary


def get_lda_representation(lda_model, dictionary, paragraphs):
    """将段落表示为 LDA 主题分布"""
    corpus = [dictionary.doc2bow(para) for para in paragraphs]
    lda_representations = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_vector = [topic[1] for topic in topic_dist]
        lda_representations.append(topic_vector)
    return np.array(lda_representations)


def cross_validate(lda_representations, labels, num_folds=10):
    """10 次交叉验证"""
    kf = KFold(n_splits=num_folds, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(lda_representations):
        X_train, X_test = lda_representations[train_index], lda_representations[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return np.mean(accuracies)


# 实验参数设置
K_values = [20, 100, 500, 1000, 3000]
T_values = [5, 10, 20, 50, 100]
units = ["word", "char"]

# 加载语料库
novels = load_corpus()

results = []
for K in K_values:
    for unit in units:
        paragraphs, labels = sample_paragraphs(novels, K=K, num_samples=1000, unit=unit)
        for T in T_values:
            print(f"当前实验参数: K={K}, unit={unit}, T={T}")
            lda_model, dictionary = train_lda(paragraphs, num_topics=T)
            lda_representations = get_lda_representation(lda_model, dictionary, paragraphs)
            accuracy = cross_validate(lda_representations, labels)
            results.append({
                "K": K,
                "unit": unit,
                "T": T,
                "accuracy": accuracy
            })
            print(f"实验完成, K={K}, unit={unit}, T={T}, 准确率: {accuracy:.4f}")

# 结果分析
results_df = pd.DataFrame(results)

# 保存结果到Excel
results_df.to_excel('jinyong_lda_results.xlsx', index=False)
print("实验结果已保存到 jinyong_lda_results.xlsx")

# 绘制任务1：主题个数T对分类性能的影响
plt.figure(figsize=(12, 6))
for unit in units:
    unit_df = results_df[results_df['unit'] == unit]
    for K in K_values:
        K_df = unit_df[unit_df['K'] == K]
        # 将 pandas.Series 转换为 numpy.ndarray
        plt.plot(K_df['T'].values, K_df['accuracy'].values, label=f'K={K}, Unit={unit}')
plt.title('主题个数(T)对分类准确率的影响')
plt.xlabel('主题个数(T)')
plt.ylabel('准确率')
plt.legend()
plt.savefig('task1_T_vs_accuracy.png')
plt.close()

# 绘制任务2：词与字单元的分类差异
plt.figure(figsize=(12, 6))
for K in K_values:
    K_df = results_df[results_df['K'] == K]
    for T in T_values:
        T_df = K_df[K_df['T'] == T]
        x_pos = np.arange(len(units)) + (T-5)*0.2  # 动态调整x轴位置
        # 将 pandas.Series 转换为 numpy.ndarray
        plt.bar(x_pos, T_df['accuracy'].values, width=0.2, label=f'K={K}, T={T}')
plt.title('词与字单元的分类准确率差异')
plt.xlabel('分析单元')
plt.ylabel('准确率')
plt.xticks(np.arange(len(units)) + (max(T_values)-5)*0.2, units)
plt.legend()
plt.savefig('task2_unit_comparison.png')
plt.close()

# 绘制任务3：段落长度K对性能的影响
plt.figure(figsize=(12, 6))
for unit in units:
    unit_df = results_df[results_df['unit'] == unit]
    for T in T_values:
        T_df = unit_df[unit_df['T'] == T]
        # 将 pandas.Series 转换为 numpy.ndarray
        plt.plot(T_df['K'].values, T_df['accuracy'].values, label=f'T={T}, Unit={unit}')
plt.title('段落长度(K)对主题模型性能的影响')
plt.xlabel('段落长度(K)')
plt.ylabel('准确率')
plt.legend()
plt.savefig('task3_K_vs_accuracy.png')
plt.close()

print("所有图表已生成并保存为PNG文件")    