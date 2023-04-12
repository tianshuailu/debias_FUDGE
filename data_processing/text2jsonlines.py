
"""
The following code combine a source text file and a target text file into a jsonlines
file that meets hugging face training data's requirement.
"""

import argparse
import jsonlines
def main():
    parser = argparse.ArgumentParser(description="Converts parallel source and text files to jsonlines.")
    parser.add_argument('--source', type=str, required=True, metavar='PATH', help='source input text, one sentence per line')
    parser.add_argument('--target', type=str, required=True, metavar='PATH', help='target input text, one sentence per line')
    parser.add_argument('--jsonlines', type=str, required=True, metavar='PATH', help='jsonlines output file, format see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation')
    parser.add_argument('--src_lang', type=str, required=True, metavar='PATH', help='source language tag')
    parser.add_argument('--trg_lang', type=str, required=True, metavar='PATH', help='target language tag')
    args = parser.parse_args()
    with open(args.source, 'r') as src_in, open(args.target, 'r') as trg_in, jsonlines.open(args.jsonlines, mode='w') as j:
        for src_line, trg_line in zip(src_in.readlines(), trg_in.readlines()):
                sample = {'translation':
                            {args.src_lang: src_line.rstrip(),
                            args.trg_lang: trg_line.rstrip()} }
                j.write(sample)
if __name__ == "__main__":
    main()
