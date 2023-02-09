import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer


def pretrain(args, model, pretrain_dataloader, path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pt)

    for epoch in range(args.epoch_pt):
        model.train()
        total_loss = 0
        for movie_id, review_meta, review_token, review_mask in tqdm(
                pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            loss = model.pre_forward(review_meta, review_token, review_mask, movie_id)
            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('[Epoch%d]\tLoss:\t%.4f' % (epoch, total_loss))

    torch.save(model.state_dict(), path)

    model.eval()
    topk = [1, 5, 10, 20]
    hit = [[], [], [], []]

    for movie_id, review_meta, review_token, review_mask in tqdm(
            pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        scores, target_id = model.pre_forward(review_meta, review_token, review_mask, movie_id, compute_score=True)
        scores = scores[:, torch.LongTensor(model.movie2ids)]

        target_id = target_id.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_id, sub_scores):
                y = model.movie2ids.index(label)
                hit[k].append(np.isin(y, score))

    print('Pre-train test done')
    for k in range(len(topk)):
        hit_score = np.mean(hit[k])
        print('[pre-train] hit@%d:\t%.4f' % (topk[k], hit_score))
