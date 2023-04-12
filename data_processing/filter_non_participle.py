import spacy
from tqdm import tqdm

nlp = spacy.load("it_core_news_sm")
"""
list = []
with open("test.it.fe.txt", "r") as f:
    for line in f:
        list.append(line.strip())

with open("test.it.fe.filter_non-participle.txt", "w") as f:
    for line in tqdm(list):
        doc = nlp(line)
        for token in doc:
            morph_list = str(token.morph).split("|")
            if ("VerbForm=Part" in morph_list and "Gender=Masc" in morph_list):
                f.write(line + "\n")
                break
"""
en_list = []
it_list = []
with open("test.en.fe.txt", "r") as f:
    for line in f:
        en_list.append(line.strip())
with open("test.it.fe.txt", "r") as f:
    for line in f:
        it_list.append(line.strip())


with open("test.it.fe.filter_non-participle.txt", "w") as it_f, open("test.en.fe.filter_non-participle.txt", "w") as en_f:
    for i, line in tqdm(enumerate(it_list)):
        doc = nlp(line)
        for token in doc:
            morph_list = str(token.morph).split("|")
            if ("VerbForm=Part" in morph_list and "Gender=Masc" in morph_list):
                it_f.write(line + "\n")
                en_f.write(en_list[i] + "\n")
                break
