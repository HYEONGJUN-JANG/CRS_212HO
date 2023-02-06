import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer


def pretrain(args, model, pretrain_dataloader, path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pt)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for epoch in range(args.epoch_pt):
        model.train()
        total_loss = 0
        total_loss_lm = 0
        for movie_id, plot_meta, plot_token, plot_mask, review_meta, review_token, review_mask in tqdm(
                pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            loss = model.pre_forward(plot_meta, plot_token, plot_mask, review_meta, review_token,
                                     review_mask, movie_id)
            # scores = scores[:, torch.LongTensor(model.movie2ids)]
            # loss = model.criterion(scores, movie_id)
            # joint_loss = loss + masked_lm_loss
            total_loss += loss.data.float()
            # total_loss_lm += masked_lm_loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('[Epoch%d]\tLoss:\t%.4f' % (epoch, total_loss))

    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식

    model.eval()
    topk = [1, 5, 10, 20]
    hit = [[], [], [], []]
    gen_resps = []
    ref_resps = []

    for movie_id, plot_meta, plot_token, plot_mask, review_meta, review_token, review_mask in tqdm(
            pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        scores, target_id = model.pre_forward(plot_meta, plot_token, plot_mask,
                                              review_meta, review_token,
                                              review_mask,
                                              movie_id,
                                              compute_score=True)

        # Moive name 예측 결과 디코딩
        # predicted_token_ids = torch.argmax(prediction_scores[:, 1:, :], dim=2)
        # context_len = torch.sum(dup_mask_label > 0, dim=1)
        # gen_resps.extend([tokenizer.decode(ids[:length]) for ids, length in zip(predicted_token_ids, context_len)])
        # ref_resps.extend([tokenizer.decode(ids[1:length + 1]) for ids, length in zip(dup_mask_label, context_len)])

        # for ref_seq, gen_seq, length in zip(dup_mask_label, gen_seqs, context_len):
        #     gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
        #     gen_resps.append(gen_seq[:length])
        #     ref_resps.append(ref_seq[1:length + 1])

        scores = scores[:, torch.LongTensor(model.movie2ids)]

        # Item에 해당하는 것만 score 추출 (실험: 학습할 때도 똑같이 해줘야 할 지?)
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

    # Movie name 예측 결과 작성
    # movie_name_path = f"./results/{path.split('.')[1].split('/')[-1]}_movie_name_pred_result.txt"
    # with open(movie_name_path, 'w', encoding='utf-8') as result_f:
    #     for ref, gen in zip(ref_resps, gen_resps):
    #         result_f.write(f"REF:\t{ref}\t/\tGEN:\t{gen}\n")
