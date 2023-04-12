"""
split the two ParlaMint Italian data sets: male.all.txt and female.all.txt into sentences

example call:
python split_utterance.py --infile male.all.txt --outfile male.split.txt
python split_utterance.py --infile female.all.txt --outfile female.split.txt

"""
from argparse import ArgumentParser
import spacy
import re

spacy_model = spacy.load("it_core_news_sm")
spacy_model.create_pipe('sentencizer')

def split(infile, outfile):

    split_sents = []
    i = 0
    with open(infile, 'r') as f:
        for line in f.readlines():
            doc = spacy_model(line)
            for sent in doc.sents:
                sentence = sent.text.strip()
                split_sents.append(sentence)

    with open(outfile, 'w') as f:
        for sent in split_sents:
            f.write(sent +"\n")



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    #parser.add_argument('--max-len-in-words', type=int, required=False, default=150, help="Maximum length for samples, if not set, will keep all. Default: 150")

    args = parser.parse_args()
    split(args.infile, args.outfile)
