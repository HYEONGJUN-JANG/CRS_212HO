import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm


def train_recommender(args, model, train_dataloader, test_dataloader, path, results_file_path, pretrain_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=args.lr_ft / args.warmup_gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=args.warmup_gamma)

    for epoch in range(args.epoch):

        model.eval()
        topk = [1, 5, 10, 20, 50]
        hit_ft = [[], [], [], [], []]
        hit_pt = [[], [], [], [], []]

        # Pre-training Test
        for movie_id, meta, plot_token, plot_mask, review_token, review_mask in tqdm(pretrain_dataloader):
            scores, target_id = model.pre_forward(meta, plot_token, plot_mask, review_token, review_mask, movie_id,
                                                  compute_score=True)
            scores = scores[:, torch.LongTensor(model.movie2ids)]

            # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
            target_id = target_id.cpu().numpy()

            for k in range(len(topk)):
                sub_scores = scores.topk(topk[k])[1]
                sub_scores = sub_scores.cpu().numpy()

                for (label, score) in zip(target_id, sub_scores):
                    y = model.movie2ids.index(label)
                    hit_pt[k].append(np.isin(y, score))

        print('Epoch %d : pre-train test done' % (epoch))
        for k in range(len(topk)):
            hit_score = np.mean(hit_pt[k])
            print('[pre-train] hit@%d:\t%.4f' % (topk[k], hit_score))

        with open(results_file_path, 'a', encoding='utf-8') as result_f:
            result_f.write(
                '[PRE TRAINING] Epoch:\t%d\tH@1\t%.4f\tH@5\t%.4f\tH@10\t%.4f\tH@20\t%.4f\tH@50\t%.4f\n' % (
                    epoch, np.mean(hit_pt[0]), np.mean(hit_pt[1]), np.mean(hit_pt[2]), np.mean(hit_pt[3]),
                    np.mean(hit_pt[4])))

        # Fine-tuning Test
        for batch in test_dataloader.get_rec_data(args.batch_size, shuffle=False):
            context_entities, context_tokens, _, _, _, _, _, target_items = batch

            scores = model.forward(context_entities, context_tokens)

            # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
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
                '[FINE TUNING] Epoch:\t%d\tH@1\t%.4f\tH@5\t%.4f\tH@10\t%.4f\tH@20\t%.4f\tH@50\t%.4f\n' % (
                    epoch, np.mean(hit_ft[0]), np.mean(hit_ft[1]), np.mean(hit_ft[2]), np.mean(hit_ft[3]),
                    np.mean(hit_ft[4])))

        # TRAIN
        model.train()
        total_loss = 0

        logger.info(f'[Recommendation epoch {str(epoch)}]')
        logger.info('[Train]')

        for batch in train_dataloader.get_rec_data(args.batch_size):
            context_entities, context_tokens, meta, plot, plot_mask, review, review_mask, target_items = batch
            scores_ft = model.forward(context_entities, context_tokens)
            loss_ft = model.criterion(scores_ft, target_items.to(args.device_id))

            loss_pt = model.pre_forward(meta, plot, plot_mask, review, review_mask, target_items)
            # loss_pt = model.criterion(scores_pt, target_items.to(args.device_id))

            loss = loss_ft + (loss_pt * args.loss_lambda)
            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss:\t%.4f\t%f' % (total_loss, scheduler.get_last_lr()[0]))
        # scheduler.step()

    model.eval()
    topk = [1, 5, 10, 20, 50]
    hit_ft = [[], [], [], [], []]
    hit_pt = [[], [], [], [], []]

    # Pre-training Test
    for movie_id, meta, plot_token, plot_mask, review_token, review_mask in tqdm(pretrain_dataloader):
        scores, target_id = model.pre_forward(meta, plot_token, plot_mask, review_token, review_mask, movie_id,
                                              compute_score=True)
        scores = scores[:, torch.LongTensor(model.movie2ids)]

        # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
        target_id = target_id.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_id, sub_scores):
                y = model.movie2ids.index(label)
                hit_pt[k].append(np.isin(y, score))

    print('Epoch %d : pre-train test done' % (args.epoch))
    for k in range(len(topk)):
        hit_score = np.mean(hit_pt[k])
        print('[pre-train] hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[PRE TRAINING] Epoch:\t%d\tH@1\t%.4f\tH@5\t%.4f\tH@10\t%.4f\tH@20\t%.4f\tH@50\t%.4f\n' % (
                args.epoch, np.mean(hit_pt[0]), np.mean(hit_pt[1]), np.mean(hit_pt[2]), np.mean(hit_pt[3]),
                np.mean(hit_pt[4])))

    # Fine-tuning Test
    for batch in test_dataloader.get_rec_data(args.batch_size, shuffle=False):
        context_entities, context_tokens, _, _, _, _, target_items = batch
        scores = model.forward(context_entities, context_tokens)

        # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
        scores = scores[:, torch.LongTensor(model.movie2ids)]
        target_items = target_items.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_items, sub_scores):
                target_idx = model.movie2ids.index(label)
                hit_ft[k].append(np.isin(target_idx, score))

    print('Epoch %d : test done' % (args.epoch))

    for k in range(len(topk)):
        hit_score = np.mean(hit_ft[k])
        print('hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[FINE TUNING] Epoch:\t%d\tH@1\t%.4f\tH@5\t%.4f\tH@10\t%.4f\tH@20\t%.4f\tH@50\t%.4f\n' % (
                args.epoch, np.mean(hit_ft[0]), np.mean(hit_ft[1]), np.mean(hit_ft[2]), np.mean(hit_ft[3]),
                np.mean(hit_ft[4])))

    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식

# todo: train_generator
