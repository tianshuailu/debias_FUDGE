"""
remove the sentences that appear in both female sentences and male sentences

example call:
python deduplicate.py --female-in female.split.txt --male-in male.split.txt --female-out fo.txt --male-out mo.txt

"""

from argparse import ArgumentParser
import re

def split(female_in, male_in, female_out, male_out):

    dedup_sents_female = []
    dedup_sents_male = []
    sents_female = dict()
    sents_male = dict()


def split(female_in, male_in, female_out, male_out):

    dedup_sents_female = []
    dedup_sents_male = []
    sents_female = dict()
    sents_male = dict()


    with open(female_in, 'r') as f:
        for line in f.readlines():
            sents_female[line.strip()] = 1
    with open(male_in, "r") as m:
        for line in m.readlines():
            sents_male[line.strip()] = 1

    for line in sents_female.keys():
        if line not in sents_male:
            dedup_sents_female.append(line)

    for line in sents_male.keys():
        if line not in sents_female:
            dedup_sents_male.append(line)

    with open(female_out, 'w') as f:
        for sent in dedup_sents_female[:-1]:
            if len(sent.split()) >= 3 and len(sent.split()) <= 200:
                f.write(sent + "\n")
        if len(dedup_sents_female[-1].split()) >= 3 and len(dedup_sents_female[-1].split()) <= 200:
            f.write(dedup_sents_female[-1])

    with open(male_out, 'w') as f:
        for sent in dedup_sents_male[:-1]:
            if len(sent.split()) >= 3 and len(sent.split()) <= 200:
                f.write(sent + "\n")
        if len(dedup_sents_male[-1].split()) >= 3 and len(dedup_sents_male[-1].split()) <= 200:
            f.write(dedup_sents_male[-1])


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--female-in', type=str, required=True)
    parser.add_argument('--male-in', type=str, required=True)
    parser.add_argument('--female-out', type=str, required=True)
    parser.add_argument('--male-out', type=str, required=True)

    args = parser.parse_args()
    split(args.female_in, args.male_in, args.female_out, args.male_out)
