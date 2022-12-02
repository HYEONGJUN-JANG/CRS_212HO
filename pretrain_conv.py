import math

import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import json
from model import Projector


def evaluate(titles, response, preds, tokenizer, log=False, log_file_path=None):
    log_file = open(log_file_path, 'a', buffering=1, encoding='utf-8')

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    decoded_responses = tokenizer.batch_decode(response, skip_special_tokens=False)
    decoded_responses = [decoded_response.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_response in
                         decoded_responses]
    decoded_responses = [response.strip() for response in decoded_responses]

    decoded_titles = tokenizer.batch_decode(titles, skip_special_tokens=False)
    decoded_titles = [decoded_title.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_title in
                      decoded_titles]
    decoded_titles = [title.strip() for title in decoded_titles]

    if log:
        for response, pred, title in zip(decoded_responses, decoded_preds, decoded_titles):
            log_file.write(json.dumps({
                'pred': pred,
                'label': title + ' ' + response
            }, ensure_ascii=False) + '\n')


def pretrain_conv(args, model, gpt_model, gpt_config, tokenizer_gpt, pretrain_dataloader, pretrain_dataloader_test,
                  path=None,
                  save_path=None):
    modules = [gpt_model]
    no_decay = ["bias", "LayerNorm.weight"]
    projector = Projector(gpt_config.hidden_size, args.kg_emb_dim).to(args.device_id)

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
        },
        {
            "params": projector.parameters()
        }
    ]

    num_update_steps_per_epoch = math.ceil(len(pretrain_dataloader))
    max_train_steps = args.conv_epoch_ft * num_update_steps_per_epoch

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.conv_lr_pt)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_train_steps)

    # train
    for epoch in range(args.conv_epoch_pt):
        logger.info(f'[Conv - Pre-training] Train-{epoch}')
        total_loss = 0
        gpt_model.train()
        projector.train()
        for batch in tqdm(pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            with torch.no_grad():
                entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask = model.get_representations(
                    batch['context_entities'], batch['context_bert'].input_ids)

            encode_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                                                   entity_padding_mask)

            loss = gpt_model(**batch['context'], labels=batch['response'], encoder_hidden_states=None,
                             encoder_attention_mask=None).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.data.float()
        print('[Epoch%d]\tLoss:\t%.4f' % (epoch, total_loss))

    # test
    logger.info('[Conv - Pre-training] Test')
    gpt_model.eval()
    projector.eval()
    for batch in tqdm(pretrain_dataloader_test, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        with torch.no_grad():
            entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask = model.get_representations(
                batch['context_entities'], batch['context_bert'].input_ids)

        encode_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                                               entity_padding_mask)

        gen_seqs = gpt_model.generate(**batch['context'], encoder_hidden_states=None,
                                      encoder_attention_mask=None,
                                      max_new_tokens=args.max_gen_len)
        gen_resp_ids = []
        for gen_seq, length in zip(gen_seqs, batch['context_len']):
            gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer_gpt.pad_token_id]
            gen_resp_ids.append(gen_seq)
        evaluate(batch['context'].input_ids, batch['response'], gen_resp_ids, tokenizer_gpt,
                 log=True, log_file_path=path)

        if save_path is not None:
            torch.save(gpt_model.state_dict(), save_path)  # TIME_MODELNAME 형식
