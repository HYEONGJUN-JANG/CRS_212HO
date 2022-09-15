import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm


def pretrain(args, model, pretrain_dataloader, path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1):
        model.train()
        total_loss = 0

        for movie_id, plot_token, plot_mask, review_token, review_mask in tqdm(
                pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            scores = model.pretrain(plot_token, plot_mask, review_token, review_mask)
            scores = scores[:, torch.LongTensor(model.movie2ids)]

            loss = model.criterion(scores, movie_id)
            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss:\t%.4f' % total_loss)

    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식
