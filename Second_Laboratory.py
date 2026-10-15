import gensim

word1 = "бульвар_NOUN"
word2 = "прохожий_NOUN"

word2vec = gensim.models.KeyedVectors.load_word2vec_format('cbow.txt', binary=False)

res = [word for word, _ in word2vec.most_similar(positive=["переулок_NOUN", "незнакомец_NOUN"], topn=30)][:10]
if word1 in res and word2 in res:
    print(f"бульвар_NOUN и прохожий_NOUN находятся в топ-10 ближайших соседей к переулок_NOUN и незнакомец_NOUN в индексах {res.index(word1)} и {res.index(word2)}")
