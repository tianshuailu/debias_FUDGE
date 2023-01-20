"""
Example call:

    python simplification_evaluation.py \
        --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_test.tsv \
        --hyp_file /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v4_test_lr0.47_ls0.79_wr0.43_td0.42.pred \
        --compute_ppl

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
