# Debias_FUDGE

This repository code for applying [FUDGE](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation) to reduce gender bias in NMT. Many of the scripts were also adapted from [SimpleFUDGE](https://github.com/ZurichNLP/SimpleFUDGE).

To setup the environment, please refer to [FUDGE](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation).

## Data Preprocessing

### Europarl-Speaker-Information
To process the data for training the baseline, we first use
```
./data_processing/text2jsonlines.py
```
to combine the English source text file and the Italian source text file into a jsonlines file.

### ParlaMint 2.1
To process the data for training the classifiers, we first use
```
./data_processing/split_utterance.py
```
to split the utterances into sentences, then use
```
./data_processing/deduplicate.py
```
to remove the dupilcate sentences. Finally use
```
./data_processing/filter_non_adj.py
```
and
```
./data_processing/filter_non_participle.py
```
to filter out the senences that do not contain adjectives or participles.

## Baseline

First we trim mT5-small with the dictionary
```
./trim_mt5/dict_25k.txt
```
and the script
```
./trim_mt5/trim_mt5.py
```
And we also put the tokenization scripts in the same folder.

The code for training the baseline is in the following folder:
```
./baseline/run_translation_adapt.py
```
Below is an example script to train the tagged baseline:
```
python /path_of/run_translation_adapt.py \
    --model_name_or_path /path_of/mt5-small-trimmed \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang it \
    --train_file /path_to/train.json \
    --validation_file /path_to/valid.json \
    --test_file /path_to/test.json \
    --output_dir /path_of_output \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 5 \
    --save_steps 3000 \
    --eval_steps 3000 \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model bleu \
    --greater_is_better True \
    --lr_scheduler_type "cosine" \
    --gradient_accumulation_steps 8
```

## Classifier

In script `./data.py`, we define the data splits of the feminine classifier and the masculine classifier. In script `./model.py` we define the structure of the two classifiers. And in script `./main.py`, we specify the loss function and data logging settings for [Weights & Biases](https://wandb.ai/site). Finally, train the classifier with the following example command:
```
python main.py \
    --task female_classifier \
    --data_dir /path_to_data \
    --save_dir /path_of_output \
    --model_path_or_name /path_of_translation_model \
    --num_workers 12 \
    --lr 1e-3 \
    --batch_size 64 \
    --epochs 15 \
    --wandb name_of_wandb_project
```

## Decoding

For decoding, run the script `./inference.py` with the following example command:
```
python inference.py \
    --condition_model /path_of_classifier \
    --generation_model /part_of_translation_model \
    --infile /path_of_source_en_file \
    --outpath /path_of_output \
    --batch_size 1 --condition_lambda 5 \
    --max_length 512
```
And when condition_lambda = 0, the output of FUDGE is the same as the underlying translation model.

## MuST-SHE Evaluation

For both the word-level Parts-Of-Speech evaluation and the agreement level evaluation, please refer to [MuST-SHE](https://ict.fbk.eu/must-she/) for more detail.


