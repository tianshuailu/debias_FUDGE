#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Tannon Kew

Example call:

    python simplification_evaluation.py \
        --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_test.tsv \
        --hyp_file /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v4_test_lr0.47_ls0.79_wr0.43_td0.42.pred \
        --compute_ppl

"""

import argparse
import numpy as np
from tqdm import tqdm
import random
from typing import List, Optional

import pandas as pd

import torch # for bertscore
from easse import sari, bleu, fkgl, bertscore, quality_estimation # samsa fails dep: tupa

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

from perplexity import distilGPT2_perplexity_score
from distinct_n import distinct

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True, help='')
    parser.add_argument('--ref_file', type=str, required=False, help='')
    parser.add_argument('--hyp_file', type=str, required=True, help='')
    parser.add_argument('--compute_ppl', action='store_true', help='computes text ppl scores using DistilBERT. Note: this is not parallelised so takes a bit longer')
    parser.add_argument('--mode', type=str, required=False, default='not_empty', help='hypothesis selection method to use if input file contains n-best list.')
    return parser.parse_args()

def read_lines(filename):
    """from easse/utils/helpers.py"""
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines

def read_split_lines(filename, split_sep='\t'):
    """from easse/utils/helpers.py"""
    with open(filename, encoding="utf-8") as f:
        split_instances = []
        for line in f:
            split_instances.append([split.strip() for split in line.split(split_sep)])

    return split_instances

def rerank_nbest(hyp_sents, ref_sent):
    """rerank hypotheses according to semantic similarity with src sentence"""
    embs = model.encode(hyp_sents + [ref_sent], convert_to_tensor=True)
    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embs, embs)    
    index_reranking = torch.argsort(cosine_scores[-1][:-1], descending=True).cpu().tolist()
    hyp_sents[:] = [hyp_sents[i] for i in index_reranking] # rerank according to ranked indices
    return hyp_sents

def select_hyp(nbest_hyps: List[List[str]], src_sents: Optional[List[str]] = None, mode='not_empy') -> List[str]:
    """
    selects a 1-best hypothesis to use for scoring. By deafult, we take the first non-empty hypothesis!
    """
    if mode == 'sim' and src_sents is not None and len(nbest_hyps[0]) > 1:
        hyp_sents = []
        for nbest, src in zip(nbest_hyps, src_sents):
            nbest = rerank_nbest(nbest, src)
            hyp_sents.append(nbest[0])
    elif mode == 'random':
        hyp_sents = [random.choice(nbest) for nbest in nbest_hyps]
    elif mode == 'model':
        hyp_sents = [i[0] for i in nbest_hyps]
    elif mode == 'not_empty':
        hyp_sents = []
        for nbest in (nbest_hyps):
            nbest = list(filter(lambda hyp: len(hyp) > 0, nbest))
            try:
                hyp_sents.append(nbest[0])
            except:
                hyp_sents.append('')
    else:
        raise RuntimeError(f'Could not select hypothesis with mode {mode}')
    
    return hyp_sents

def ppl_score(sents):
    """computes ppl score with GPT model"""
    return np.array([distilGPT2_perplexity_score('. '+s) for s in tqdm(sents)])

if __name__ == '__main__':

    args = set_args()

    if not args.ref_file: # assumes human refs are in src file after first column
        sents = read_split_lines(args.src_file, split_sep='\t')
        src_sents = [i[0] for i in sents]
        refs_sents = [i[1:] for i in sents]
        refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose to number samples x len(test set)
    else:
        src_sents = read_lines(args.src_file)
        refs_sents = read_lines(args.ref_file)
    
    # handle n-best list generations   
    nbest_hyps = read_split_lines(args.hyp_file, split_sep='\t')
    hyp_sents = select_hyp(nbest_hyps, src_sents, mode=args.mode)
    
    assert len(hyp_sents) == len(src_sents)

    results = {'file': args.hyp_file}
    
    results['ppl'] = None
    if args.compute_ppl:
        # results['ppl_diff'] = ppl_diff(hyp_sents, refs_sents)
        results['ppl'] = ppl_score(hyp_sents).mean()

    
    if torch.cuda.is_available():
        precision_ref, recall_ref, f1_ref = bertscore.corpus_bertscore(hyp_sents, refs_sents)
        precision_src, recall_src, f1_src = bertscore.corpus_bertscore(hyp_sents, list(map(list, [*zip(*[[s] for s in src_sents])])))
    else:
        precision_ref, recall_ref, f1_ref = None, None, None
        precision_src, recall_src, f1_src = None, None, None
    
    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
    results['bertscore_p_ref'] = precision_ref * 100 if precision_ref else None
    results['bertscore_r_ref'] = recall_ref * 100 if recall_ref else None
    results['bertscore_f1_ref'] = f1_ref * 100 if f1_ref else None
    results['bertscore_p_src'] = precision_src * 100 if precision_src else None
    results['bertscore_r_src'] = recall_src * 100 if recall_src else None
    results['bertscore_f1_src'] = f1_src * 100 if f1_src else None
    
    results['intra_dist1'], results['intra_dist2'], results['inter_dist1'], results['inter_dist2'] = distinct(hyp_sents)

    qe = quality_estimation.corpus_quality_estimation(src_sents, hyp_sents)
    results.update(qe)
    
    df = pd.DataFrame(data=results, index=[0])
    print(df.to_csv(sep=';', index=False))