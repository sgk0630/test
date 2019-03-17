import jieba.analyse as analyse
import pandas as pd
import jieba
import gensim
from gensim import corpora, similarities, models

# 读取文件
df = pd.read_csv(r'.\data\technology_news.csv', encoding='utf-8')
df = df.dropna()
content = df.content.values.tolist()

# 用jieba.analyse.extract_tags提取关键词
# lines = ''.join(content)
# ana = ' '.join(analyse.extract_tags(sentence=lines, topK=20, withWeight=False, allowPOS=(), withFlag=False))
# print(type(ana))

# 读入停词表
stopwords_path = r'D:\pycharm\stopwords.txt'
stopwords = pd.read_csv(stopwords_path, sep='\t', names=['stopword'], index_col=None, quoting=3, encoding='utf-8')
stopwords = stopwords.values
print(type(stopwords))

# 分词过滤
segments = []
for line in content:
	try:
		segs = jieba.lcut(line)
		segs = filter(lambda x: len(x) > 1, segs)
		segs = filter(lambda x: x not in stopwords, segs)  # 过滤器返回值是iterator对象
		segments.append(list(segs))
	except():
		print(line)
		continue
# print(segments[:2])

# 词袋
dictionary = corpora.Dictionary(segments)
corpus = [dictionary.doc2bow(segment) for segment in segments]
# print(dictionary[20])

# 建模
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(topicno=3, topn=5))


