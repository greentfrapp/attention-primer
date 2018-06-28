import codecs
import json
import regex
from collections import Counter

def make_dict(filepath, dictpath, min_freq=10):
	text = codecs.open(filepath, 'r', 'utf-8').read().lower()
	text = regex.sub("<.*>.*</.*>\n", "", text)
	text = regex.sub("[^\s\p{Latin}']", "", text)
	words = text.split()
	word2count = Counter(words)

	dictionary = ['<PAD>', '<UNK>', '<START>', '<END>']
	for word, count in word2count.items():
		if count >= min_freq:
			dictionary.append(word)

	with open(dictpath, 'w', encoding='utf-8') as file:
		json.dump(dictionary, file)


if __name__ == "__main__":
	de_file = "data/train.tags.de-en.de"
	en_file = "data/train.tags.de-en.en"
	make_dict(de_file, "data/de_dict.json")
	make_dict(en_file, "data/en_dict.json")
	