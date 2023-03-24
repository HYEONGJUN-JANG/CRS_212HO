from math import ceil

import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

topk = [1, 5, 10, 20, 50]


def pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit):
    model.eval()
    hit_pt = [[], [], [], [], []]

    # Pre-training Test
    for movie_id, review_meta, review_token, review_mask in tqdm(
            pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        scores, target_id = model.pre_forward(review_meta, review_token,
                                              review_mask, movie_id, compute_score=True)
        scores = scores[:, torch.LongTensor(model.movie2ids)]

        target_id = target_id.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_id, sub_scores):
                y = model.movie2ids.index(label)
                hit_pt[k].append(np.isin(y, score))

    print('Epoch %d : pre-train test done' % epoch)
    for k in range(len(topk)):
        hit_score = np.mean(hit_pt[k])
        print('[pre-train] hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[PRE TRAINING] Epoch:\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                epoch, 100 * np.mean(hit_pt[0]), 100 * np.mean(hit_pt[1]), 100 * np.mean(hit_pt[2]),
                100 * np.mean(hit_pt[3]), 100 * np.mean(hit_pt[4])))

    if epoch == 0:
        content_hit[0] = 100 * np.mean(hit_pt[0])
        content_hit[1] = 100 * np.mean(hit_pt[2])
        content_hit[2] = 100 * np.mean(hit_pt[4])


def finetuning_evaluate(model, test_dataloader, epoch, results_file_path, initial_hit, best_hit, eval_metric):
    hit_ft = [[], [], [], [], []]
    # Fine-tuning Test
    for batch in test_dataloader.get_rec_data(shuffle=False):
        context_entities, context_tokens, _, _, _, target_items = batch
        scores = model.forward(context_entities, context_tokens)
        scores = scores[:, torch.LongTensor(model.movie2ids)]

        target_items = target_items.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_items, sub_scores):
                target_idx = model.movie2ids.index(label)
                hit_ft[k].append(np.isin(target_idx, score))

    print('Epoch %d : test done' % (epoch))

    for k in range(len(topk)):
        hit_score = np.mean(hit_ft[k])
        print('hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[FINE TUNING] Epoch:\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                epoch, 100 * np.mean(hit_ft[0]), 100 * np.mean(hit_ft[1]), 100 * np.mean(hit_ft[2]),
                100 * np.mean(hit_ft[3]), 100 * np.mean(hit_ft[4])))

    if epoch == 0:
        initial_hit[0] = 100 * np.mean(hit_ft[0])
        initial_hit[1] = 100 * np.mean(hit_ft[2])
        initial_hit[2] = 100 * np.mean(hit_ft[4])

    if np.mean(hit_ft[0]) > eval_metric[0]:
        eval_metric[0] = np.mean(hit_ft[0])
        for k in range(len(topk)):
            best_hit[k] = np.mean(hit_ft[k])


def train_recommender(args, model, train_dataloader, test_dataloader, path, results_file_path, pretrain_dataloader):
    best_hit = [[], [], [], [], []]
    initial_hit = [[], [], []]
    content_hit = [[], [], []]
    eval_metric = [-1]

    optimizer = optim.Adam(model.parameters(), lr=args.lr_ft)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    for epoch in range(args.epoch_ft):
        pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit)
        finetuning_evaluate(model, test_dataloader, epoch, results_file_path, initial_hit, best_hit, eval_metric)

        # TRAIN
        model.train()
        total_loss = 0

        logger.info(f'[Recommendation epoch {str(epoch)}]')
        logger.info('[Train]')

        for batch in train_dataloader.get_rec_data(args.batch_size):
            context_entities, context_tokens, review_meta, review, review_mask, target_items = batch
            scores_ft = model.forward(context_entities, context_tokens)
            loss_ft = model.criterion(scores_ft, target_items.to(args.device_id))

            loss_pt = model.pre_forward(review_meta, review, review_mask, target_items)
            loss = loss_ft + ((loss_pt) * args.loss_lambda)

            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Loss:\t%.4f\t%f' % (total_loss, scheduler.get_last_lr()[0]))
    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식

    pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit)
    finetuning_evaluate(model, test_dataloader, epoch, results_file_path, initial_hit, best_hit, eval_metric)

    best_result = [100 * best_hit[0], 100 * best_hit[2], 100 * best_hit[4]]

    return content_hit, initial_hit, best_result
