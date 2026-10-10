import gensim
import numpy as np
import itertools

word1 = "бульвар_NOUN"
word2 = "прохожий_NOUN"

word2vec = gensim.models.KeyedVectors.load_word2vec_format('cbow.txt', binary=False)

most_similar =  [word for word, score in word2vec.most_similar(positive=[word1, word2], topn=30) if word.endswith("_NOUN")]

print(len(most_similar))

combinations = list(itertools.combinations(most_similar, 2))
pairs = []

for combo1 in combinations:
    for neg in most_similar:
        if neg in combo1:
            continue    
        res = [word for word, score in word2vec.most_similar(positive = [combo1[0], combo1[1]], negative = [neg], topn=20) if word.endswith("_NOUN")][:10]
        if word1 in res and word2 in res:
            pairs.append([combo1, neg])

print(*pairs, sep="\n")