import nltk
import pymorphy3
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')

with open("Бунин Иван. Господин из Сан-Франциско.txt", "r", encoding="utf-8") as file:
    content = file.read()

def get_batches(arr, n_seqs, n_steps):
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch

    arr = arr[:n_batches * characters_per_batch]
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # пакет данных, который будет подаваться на вход сети
        x = arr[:, n:n+n_steps]
        # целевой пакет, с которым будем сравнивать предсказание, получаем сдвиганием "x" на один символ вперед
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

vocab = sorted(set(content))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in content], dtype=np.int32)

