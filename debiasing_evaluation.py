"""
Example call:

python debiasing_evaluation.py \
    --ref_file /srv/scratch1/ltian/thesis/data_wo_tag/test.it.untokenized.txt \
    --hyp_file /srv/scratch1/ltian/thesis/base_wo_tag_output/generated_predictions.txt

"""

import argparse
import evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--ref_file', type=str, required=True, help='')
    parser.add_argument('--hyp_file', type=str, required=True, help='')
    args = parser.parse_args()

    data = []
    predictions = []
    references = []

    with open(args.ref_file, 'r') as rf, open(args.hyp_file, 'r') as hf:
        for line in rf:
            data.append(line)
            references.append(data)
        for line in hf:
            predictions.append(line)

    metric = evaluate.load('sacrebleu')

    metric.add_batch(predictions=predictions, references=references)
    score = metric.compute()
    print(score)

if __name__ == "__main__":
    main()
