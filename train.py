import torch
import numpy as np
from loguru import logger
from torch import nn, optim


def train_recommender(args, model, train_dataloader, test_dataloader, path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0

        logger.info(f'[Recommendation epoch {str(epoch)}]')
        logger.info('[Train]')

        for batch in train_dataloader.get_rec_data(args.batch_size):
            context_entities, context_tokens, target_items = batch
            scores = model.forward(context_entities, context_tokens)
            # todo: 학습할 때도, item에 해당하는 것으로만 해야 할 지? 실험
            loss = model.criterion(scores, target_items.to(args.device_id))
            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss:\t%.4f' % total_loss)

        model.eval()
        topk = [1, 5, 10, 20]
        hit = [[], [], [], []]

        for batch in test_dataloader.get_rec_data(args.batch_size, shuffle=False):
            context_entities, context_tokens, target_item = batch
            scores = model.forward(context_entities, context_tokens)

            # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
            scores = scores[:, torch.LongTensor(model.movie2ids)]
            target_item = target_item.cpu().numpy()

            for k in range(len(topk)):
                sub_scores = scores.topk(topk[k])[1]
                sub_scores = sub_scores.cpu().numpy()

                for (label, score) in zip(target_item, sub_scores):
                    target_idx = model.movie2ids.index(target_item[0])
                    hit[k].append(np.isin(target_idx, score))

        print('Epoch %d : test done' % (epoch + 1))
        for k in range(len(topk)):
            hit_score = np.mean(hit[k])
            print('hit@%d:\t%.4f' % (topk[k], hit_score))

    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식

# todo: train_generator
