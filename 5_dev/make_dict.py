import codecs
import json
import regex
from collections import Counter

min_freq = 10

de_file = "de-en/train.tags.de-en.de"
en_file = "de-en/train.tags.de-en.en"

text = codecs.open(de_file, 'r', 'utf-8').read().lower()
text = regex.sub("<.*>.*</.*>\n", "", text)
text = regex.sub("[^\s\p{Latin}']", "", text)
words = text.split()
word2count = Counter(words)

dictionary = ['<PAD>', '<UNK>', '<START>', '<END>']
for word, count in word2count.items():
	if count >= min_freq:
		dictionary.append(word)

with open("de_dict.json", 'w', encoding='utf-8') as file:
	json.dump(dictionary, file)
	