import gensim
from gensim import corpora
# Create data
import pdb

documents = [['apple', 'banana', 'fruits'], ['bought', 'bicycle', 'recently', 'less', 'two', 'years', 'buy', 'bike'],
             ['colour', 'apple', 'bicycle', 'red']]
# token2id {'apple': 0, 'banana': 1, 'fruits': 2, 'bicycle': 3, 'bike': 4, 'bought': 5, 'buy': 6, 'less': 7, 'recently': 8, 'two': 9, 'years': 10, 'colour': 11, 'red': 12}
mapping = corpora.Dictionary(documents)
# 每篇文档中出现的词频率进行排序
data = [mapping.doc2bow(word) for word in documents]
# Print data
print("data:", data)
# Train LDA model
ldamodel = gensim.models.ldamodel.LdaModel(data, num_topics=2, id2word=mapping, passes=15)

# Show topics
pdb.set_trace()
topics = ldamodel.show_topics()

print(topics)

# Distribution of topics for the first document
print(ldamodel.get_document_topics(data[0]))
