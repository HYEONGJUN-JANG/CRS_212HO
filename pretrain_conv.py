import math

import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup


def pretrain_conv(args, gpt_model, pretrain_dataloader, path=None):
    modules = [gpt_model]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        }
    ]

    num_update_steps_per_epoch = math.ceil(len(pretrain_dataloader))
    max_train_steps = args.conv_epoch_ft * num_update_steps_per_epoch

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.conv_lr_ft)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_train_steps)

    for epoch in range(args.epoch_pt):
        gpt_model.train()
        total_loss = 0
        for text, mask in tqdm(pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            text = text.to(args.device_id)
            mask = mask.to(args.device_id)
            label = mask * text + ((1 - mask) * -100)
            loss = gpt_model(input_ids=text, attention_mask=mask, labels=label).loss
            # loss = gpt_model(**batch['context'], labels=batch['response']).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.data.float()
        print('[Epoch%d]\tLoss:\t%.4f' % (epoch, total_loss))

    if path is not None:
        torch.save(gpt_model.state_dict(), path)  # TIME_MODELNAME 형식
