import math

import torch
from loguru import logger
from tqdm import tqdm

from model import Projector
from evaluate_conv import ConvEvaluator

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BartModel, BartTokenizer, AdamW, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM
import json
import os

entity2id = json.load(
    open(os.path.join('data/redial', 'entity2id.json'), 'r', encoding='utf-8'))
id2entity = {idx: entity for entity, idx in entity2id.items()}


def finetuning_evaluate(args, evaluator, epoch, test_gen_dataloader, model, projector, gpt_model, tokenizer_gpt,
                        total_report):
    gpt_model.eval()
    projector.eval()
    model.eval()

    evaluator.log_file.write(f'\n*** test-{epoch} ***\n\n')
    for batches in tqdm(test_gen_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        batch = batches[0]

        with torch.no_grad():
            model_scores = model(batch['context_entities'],
                                 batch['context_bert'].input_ids)  # context_entities, context_tokens
            recommended_items = [id2entity[top1odx.item()] for top1odx in
                                 torch.topk(model_scores, 3, dim=1).indices.view(-1)]
            entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask, user_representation = model.get_representationsWithUser(
                batch['context_entities'], batch['context_bert'].input_ids)

            encoder_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                                                    entity_padding_mask, user_representation)

            gen_seqs = gpt_model.generate(**batch['context'], prompt_embeds=encoder_state,
                                          max_new_tokens=args.max_gen_len,
                                          no_repeat_ngram_size=3)

            gen_resp_ids = []
            for gen_seq, length in zip(gen_seqs, batch['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer_gpt.pad_token_id]
                gen_resp_ids.append(gen_seq[length:])
            evaluator.evaluate(gen_resp_ids, batch['response'], batch['context'], recommended_items, log=True)
    # metric
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'test/{k}'] = v

    test_report['epoch'] = epoch
    logger.info(test_report)
    total_report.append(test_report)
    evaluator.reset_metric()


def train_conversation(args, model, train_dataloader, test_gen_dataloader, gpt_model, gpt_config, tokenizer_gpt,
                       conv_results_file_path):
    total_report = []

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    max_train_steps = args.conv_epoch_ft * num_update_steps_per_epoch
    projector = Projector(gpt_config, model.bert_config.hidden_size, args.kg_emb_dim, args.projection_order,
                          args.device_id).to(args.device_id)

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
        },
        {
            "params": projector.parameters()
        }

    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.conv_lr_ft)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_train_steps)

    evaluator = ConvEvaluator(tokenizer=tokenizer_gpt, log_file_path=conv_results_file_path)

    # train loop
    finetuning_evaluate(args, evaluator, 0, test_gen_dataloader, model, projector, gpt_model, tokenizer_gpt,
                        total_report)
    for epoch in range(args.conv_epoch_ft):
        logger.info(f'[Conversation epoch {str(epoch)}]')
        logger.info('[Train]')
        total_loss = 0
        gpt_model.train()
        projector.train()
        model.eval()
        for step, batches in enumerate(tqdm(train_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')):
            batch = batches[0]
            pre_batch = batches[1]
            with torch.no_grad():
                entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask, user_representation = model.get_representationsWithUser(
                    batch['context_entities'], batch['context_bert'].input_ids)
                pre_entity_representations, pre_entity_padding_mask, pre_kg_embedding, pre_token_embedding, pre_token_padding_mask, user_representation = model.get_representationsWithUser(
                    pre_batch['context_entities'], pre_batch['context_bert'].input_ids)

            encoder_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                                                    entity_padding_mask, user_representation)

            pre_encoder_state, pre_encoder_mask = projector(pre_token_embedding, pre_token_padding_mask,
                                                            pre_entity_representations,
                                                            pre_entity_padding_mask, user_representation)

            loss_ft = gpt_model(**batch['context'], conv_labels=batch['response'], prompt_embeds=encoder_state,
                                conv=True).conv_loss
            loss_pt = gpt_model(**pre_batch['context'], conv_labels=pre_batch['response'], conv=True,
                                prompt_embeds=pre_encoder_state).conv_loss

            loss = loss_ft + ((loss_pt) * args.conv_loss_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.data.float()
        print('Total Loss:\t%.4f' % total_loss)
        print('Loss_pt:\t%.4f\t\t Loss_ft:\t%.4f' % (loss_pt, loss_ft))

        logger.info('[Test]')
        finetuning_evaluate(args, evaluator, epoch, test_gen_dataloader, model, projector, gpt_model, tokenizer_gpt,
                            total_report)
