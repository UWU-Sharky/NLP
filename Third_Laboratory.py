from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F

import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased")

model.eval()

word1 = "происходит"
word2 = "отличается"

text = f"В этой лаборатории [MASK] что-то странное, и каждый новый эксперимент сильно [MASK] от предыдущего."

inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

mask_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]

answer = []
softmax = F.softmax(logits, dim = -1)

for i, idx in enumerate(mask_indices):
    mask_words = softmax[0, idx, :]
    top_results = torch.topk(mask_words, k=10)

    print(f"Топ 10 предсказаний для маски #{i + 1} (позиция {idx.item()}):")
    
    current = []
    
    for score, token_id in zip(top_results.values, top_results.indices):
        word = tokenizer.decode([token_id]).strip()
        percentage = score.item() * 100
        current.append((word,percentage))
        print(f"    - {word} ({percentage:.2f}%)")

    answer.append(current)

for mask_index, res in enumerate(answer):
    res_dict = dict(res) 
    
    if word1 in res_dict and res_dict[word1] > 10:
        print(f"слово {word1} входит в маску #{mask_index + 1} с вероятностью {res_dict[word1]:.2f}%")
        
    if word2 in res_dict and res_dict[word2] > 10:
        print(f"слово {word2} входит в маску #{mask_index + 1} с вероятностью {res_dict[word2]:.2f}%")
