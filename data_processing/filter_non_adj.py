import spacy

nlp = spacy.load("it_core_news_sm")

with open("fo_5-200.txt", "r") as f:
    for line in f:
        list.append(line)

with open("fo_5-200_filter.txt", "w") as f:
    for line in list:
        doc = nlp(line)
        for token in doc:
            if token.pos_ == "ADJ":
                f.write(line + "\n")
                break
