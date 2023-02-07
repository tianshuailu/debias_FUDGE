import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support


from data import Dataset
# from data_v2 import SimplificationDataset as Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *

import wandb

def train(model, dataset, optimizer, criterion, epoch, args, data_start_index, logger=None):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index + args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', num_workers=args.num_workers, indices=list(range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')

    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        inputs, lengths, future_words, log_probs, labels, classification_targets, syllables_to_go, future_word_num_syllables, rhyme_group_index = batch
        if args.task not in ['formality', 'iambic', 'simplify', 'male_classifier', 'female_classifier']:
            if not args.debug and len(inputs) != args.batch_size: # it'll screw up the bias...?
                continue
        scores = model(inputs, lengths, future_words, log_probs, syllables_to_go, future_word_num_syllables, rhyme_group_index, run_classifier=True)
        if args.task in ['formality', 'simplify', 'male_classifier', 'female_classifier']: # we're learning for all positions at once. scores are batch x seq
            expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1]) # batch x seq
            length_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            loss = criterion(scores.flatten()[length_mask.flatten()==1], expanded_labels.flatten().float()[length_mask.flatten()==1])
        elif args.task in ['iambic', 'newline']:
            use_indices = classification_targets.flatten() != -1
            loss = criterion(scores.flatten()[use_indices], classification_targets.flatten().float()[use_indices])
        else: # topic, rhyme
            loss = criterion(scores.flatten(), labels.flatten().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.detach(), len(labels))
        if logger is not None:
            logger.log({'train_loss': loss})
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return data_start_index


def validate(model, dataset, criterion, epoch, args, logger, threshold=0.0):
    model.eval()
    random.seed(0)
    loader = dataset.loader('val', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Validation: ')
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            batch = [tensor.to(args.device) for tensor in batch]
            inputs, lengths, future_words, log_probs, labels, classification_targets, syllables_to_go, future_word_num_syllables, rhyme_group_index = batch
            if args.task not in ['formality', 'iambic', 'simplify', 'male_classifier', 'female_classifier']: # topic predictor
                if not args.debug and len(inputs) != args.batch_size:
                    continue
            scores = model(inputs, lengths, future_words, log_probs, syllables_to_go, future_word_num_syllables, rhyme_group_index, run_classifier=True)
            if args.task in ['formality', 'simplify', 'male_classifier', 'female_classifier']: # we're learning for all positions at once. scores are batch x seq
                expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1]) # batch x seq
                length_mask = pad_mask(lengths).permute(1, 0) # batch x seq
                #m = nn.Sigmoid()
                loss = criterion(scores.flatten()[length_mask.flatten()==1], expanded_labels.flatten().float()[length_mask.flatten()==1])
                # report prediction accuracy
                # https://discuss.pytorch.org/t/when-do-i-turn-prediction-numbers-into-1-and-0-for-binary-classification/130075
                # acc = ((scores.flatten()[length_mask.flatten()==1] > threshold) == expanded_labels.flatten().float()[length_mask.flatten()==1]).float().mean()
            elif args.task in ['iambic', 'newline']:
                use_indices = classification_targets.flatten() != -1
                loss = criterion(scores.flatten()[use_indices], classification_targets.flatten().float()[use_indices])
            else: # topic, rhyme
                loss = criterion(scores.flatten(), labels.flatten().float())
            loss_meter.update(loss.detach(), len(labels))

            all_scores.append(scores.flatten()[length_mask.flatten()==1].cpu())
            all_labels.append(expanded_labels.flatten().float()[length_mask.flatten()==1].cpu())

    progress.display(total_length)

    all_scores = np.concatenate(all_scores, 0)
    all_labels = np.concatenate(all_labels, 0)
    pos_label_ratio = all_labels.sum()/len(all_labels)
    auc_score = roc_auc_score(all_labels, all_scores)
    all_preds = (all_scores > threshold).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    if logger is not None:
        logger.log({
            'val_loss': loss_meter.avg,
            'val_acc': acc,
            'val_prec': prec,
            'val_rec': rec,
            'val_f1': f1,
            'val_auc': auc_score,
            'pos_label_ratio': pos_label_ratio,
            })


    return loss_meter.avg, acc, prec, rec, f1, auc_score, pos_label_ratio


def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.task == 'rhyme':
        with open(os.path.join(args.save_dir, 'rhyme_info'), 'wb') as wf:
            pickle.dump(dataset.rhyme_info, wf)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        # model = Model(model_args, 65001, len(dataset.index2word), rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None) #
        model = Model(
            model_args,
            dataset.gpt_pad_id,
            len(dataset.index2word) if args.task not in ['simplify', 'male_classifier', 'female_classifier'] else dataset.tokenizer.vocab_size,
            rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None
            ) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        # NOTE: just import pdb after loading the model here if you want to play with it, it's easy
        # model.eval()
        # import pdb; pdb.set_trace()
    else:
        # For BART models, dataset.gpt_pad_id = bart's
        # pad_token_id (assigned in data.py). Therefore we
        # pass the true vocab size of BART's tokenizer to
        # the model to construct the embedding layer.

        # model = Model(args, dataset.gpt_pad_id, len(dataset.index2word), rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None, glove_embeddings=dataset.glove_embeddings)
        model = Model(
            args,
            dataset.gpt_pad_id,
            len(dataset.index2word) if args.task not in ['simplify', 'male_classifier', 'female_classifier'] else dataset.tokenizer.vocab_size,
            rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None,
            glove_embeddings=dataset.glove_embeddings
            )
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_metric = 1e8 # lower is better for BCE
        data_start_index = 0
    print('num params', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    #criterion = nn.BCELoss().to(args.device)

    # initialise logger through wandb if project space
    # proand not a training run
    logger = wandb.init(project=args.wandb) if (args.wandb is not None) and (not args.evaluate) else None
    if logger is not None:
        wandb.config.update(args)

    if args.evaluate:
        epoch = 0
        loss, acc, prec, rec, f1, auc, label_balance = validate(model, dataset, criterion, epoch, args, logger)
        print(f'{time.ctime()} | EVALUATION | Epoch {epoch} | loss {loss:.4f} | acc {acc:.4f} | prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f} | auc = {auc:.4f} | pos ratio {label_balance:.4f}')

        return
    for epoch in range(args.epochs):
        print(f'{time.ctime()} | TRAINING | Epoch {epoch} |')
        data_start_index = train(model, dataset, optimizer, criterion, epoch, args, data_start_index, logger)
        if epoch % args.validation_freq == 0:
            loss, acc, prec, rec, f1, auc, label_balance = validate(model, dataset, criterion, epoch, args, logger)
            print(f'{time.ctime()} | EVALUATION | Epoch {epoch} | loss {loss:.4f} | acc {acc:.4f} | prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f} | auc = {auc:.4f} | pos ratio {label_balance:.4f}')

            if not args.debug:
                if loss < best_val_metric:
                    print('new best val metric', loss)
                    best_val_metric = loss
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_metric': best_val_metric,
                        'optimizer': optimizer.state_dict(),
                        'data_start_index': data_start_index,
                        'args': args
                    }, os.path.join(args.save_dir, 'model_best.pth.tar'))
                # don't save all checkpoints!
                # save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': model.state_dict(),
                #     'best_metric': metric,
                #     'optimizer': optimizer.state_dict(),
                #     'data_start_index': data_start_index,
                #     'args': args
                # }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))

    print(f'{time.ctime()} | COMPLETED TRAINING |')

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--task', type=str, required=True, choices=['iambic', 'rhyme', 'newline', 'topic', 'formality', 'simplify', 'male_classifier', 'female_classifier'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--glove', type=str, help='glove embedding init, for topic task or `True` for loading embeds from gensim for simplification')

    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')
    parser.add_argument('--dataset_info', type=str, help='saved dataset info')
    parser.add_argument('--rhyme_info', type=str, help='saved dataset rhyme info, for a ckpt with task==rhyme')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--epoch_max_len', type=int, default=None, help='max batches per epoch if set, for more frequent validation')
    parser.add_argument('--validation_freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--compute_auc', type=bool, default=True)
    parser.add_argument('--bidirectional', type=bool, default=False, help='whether or not LSTM conditioning odel is bidirectional or causal')

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=1000000, help='how often to print metrics (every X batches)')

    # added for ATS
    parser.add_argument('--model_path_or_name', type=str, default=None, help='pre-trained/fine-tuned model directory with tokenizer to use.')
    parser.add_argument('--tgt_level', type=str, default="4", help='simplification level corresponding to newsela metadata')
    parser.add_argument('--use_line_parts', action='store_true', default=False, help='whether or not to train FUDGE on constituent word sequences')
    parser.add_argument('--wandb', type=str, default=None, help='wandb project space for logging')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None

    main(args)
