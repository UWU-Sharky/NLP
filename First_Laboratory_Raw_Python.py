import nltk
import pymorphy3
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

with open("Набоков В.В.-Бледный огонь.Роман-(Набоковский корпус)-2022.txt", "r", encoding="utf-8") as file:
    content = file.read()
print(content)

sequence_of_pairs = []

morph = pymorphy3.MorphAnalyzer()

sentences = sent_tokenize(content)
for sentence in sentences:
  words = word_tokenize(sentence)

  for previous_word, current_word in zip(words, words[1:]):

    previous_analysis = morph.parse(previous_word)
    current_analysis = morph.parse(current_word)

    for first_analysis in previous_analysis:
      flag = False
      for second_analysis in current_analysis:
          if (first_analysis.tag.POS in ("NOUN", "ADJF")) and (second_analysis.tag.POS in ("NOUN", "ADJF")):
             
            same_number = first_analysis.tag.number == second_analysis.tag.number
            same_case = first_analysis.tag.case == second_analysis.tag.case

            same_gender = True
            if first_analysis.tag.number == 'sing' and second_analysis.tag.number == 'sing':
                same_gender = first_analysis.tag.gender == second_analysis.tag.gender

            if same_number and same_gender and same_case:
                sequence_of_pairs.append((first_analysis.normal_form, second_analysis.normal_form))
                flag = True
                break
      if flag:
          break

print(f"Количество пар слов до фильтрации по score {len(sequence_of_pairs)}")

min_score = 0.5
filtired_sequence = []
for pair in sequence_of_pairs:
    if pair[0].score >= min_score or pair[1].score >= min_score:
        normalized_pair = (pair[0].normal_form, pair[1].normal_form)
        filtired_sequence.append(normalized_pair)

print(f"Количество пар слов после фильтрации по score {len(filtired_sequence)}")

print(*filtired_sequence, sep='\n')