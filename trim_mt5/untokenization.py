"""
untokenize space separated sentences with MosesDetokenizer

Exmple call:
python untokenization.py \
    --tokenized_file /srv/scratch1/ltian/thesis/data_wo_tag/test.it.txt \
    --output_file_name /srv/scratch1/ltian/thesis/data_wo_tag/test.it.untokenized.txt \
    --language it
"""
import argparse
from mosestokenizer import *

def reverse_tokenization(
    tokenized_file,
    output_file_name,
    language
):
    data = []
    token_list = []
    with open(tokenized_file, "r") as mf:
        for line in mf:
            for token in line.split():
                token_list.append(token)
            data.append(token_list)
            token_list = []

    detokenized_data = []
    with MosesDetokenizer(language) as detokenize:
        for line in data:
            sentence = detokenize(line)
            detokenized_data.append(sentence)

    with open(output_file_name, "w") as mf:
        for line in detokenized_data[:-1]:
            mf.write(line + "\n")
        mf.write(detokenized_data[-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tokenized_file',
        type=str,
        required=True,
        help='The name or path to the space-separated tokenized dataset'
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        required=True,
        help='The name of the reversed tokenization dataset'
    )
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        help='The language of the dataset'
    )
    args = parser.parse_args()
    reverse_tokenization(
        tokenized_file = args.tokenized_file,
        output_file_name = args.output_file_name,
        language = args.language
    )

if __name__ == "__main__":
    main()
