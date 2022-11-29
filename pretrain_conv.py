import math

import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import json


def evaluate(self, preds, epoch, log=False, log_file_path=None):
    log_file = open(log_file_path, 'w', buffering=1)
    log_file.write(f'\n*** test-{epoch + 1} ***\n\n')
    decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    if log and hasattr(self, 'log_file'):
        for pred in decoded_preds:
            self.log_file.write(json.dumps({
                # 'context': context,
                'pred': pred
            }, ensure_ascii=False) + '\n')


def pretrain_conv(args, gpt_model, tokenizer_gpt, pretrain_dataloader, path=None):
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

    for epoch in range(args.conv_epoch_pt):
        logger.info(f'[Pre-training] Train-{epoch}')
        gpt_model.train()
        total_loss = 0
        # train
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

        # test
        # logger.info(f'[Pre-training] Test-{epoch}')
        # gpt_model.eval()
        # for text, mask in tqdm(pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        #     text = text.to(args.device_id)
        #     mask = mask.to(args.device_id)
        #     # TODO: input what?
        #     gen_seqs = gpt_model.generate(text, encoder_hidden_states=None,
        #                                   encoder_attention_mask=None,
        #                                   max_new_tokens=args.max_gen_len)
        #     gen_resp_ids = []
        #     for gen_seq in gen_seqs:
        #         gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer_gpt.pad_token_id]
        #         # gen_resp_ids.append(gen_seq[length:])
        #     evaluate(gen_resp_ids, epoch, log=True, log_file_path=path)

    if path is not None:
        torch.save(gpt_model.state_dict(), path)  # TIME_MODELNAME 형식
