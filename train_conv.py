import math

import torch
from loguru import logger
from tqdm import tqdm

from model import Projector
from evaluate_conv import ConvEvaluator

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BartModel, BartTokenizer, AdamW, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM, LogitsProcessorList
import json
import os

# entity2id = json.load(
#     open(os.path.join('data/redial', 'entity2id.json'), 'r', encoding='utf-8'))
# id2entity = {idx: entity for entity, idx in entity2id.items()}
movie2name = json.load(open('data/redial/movie2name.json', 'r', encoding='utf-8'))
movieidx2name = {idx: name for key, (idx, name) in movie2name.items()}


def recommend_top1_item(batch, model):
    movie_recommended_items = []
    model_scores = model(batch['context_entities'],
                         batch['context_bert'].input_ids)  # context_entities, context_tokens
    model_scores = model_scores[:, torch.LongTensor(model.movie2ids)]

    # recommended_items = [id2entity[top1odx.item()] for top1odx in
    #                      torch.topk(model_scores, 3, dim=1).indices.view(-1)]
    top3items = torch.topk(model_scores, k=3, dim=1).indices.view(-1, 3).tolist()
    recommended_items = [[model.movie2ids[item] for top3item in top3items for item in top3item]]
    # recommended_items = [model.movie2ids[top3] for top3 in torch.topk(model_scores, 3, dim=1).indices.view(-1, 3).tolist()]
    for items in recommended_items:
        movie_recommended_items.append([movieidx2name[item] for item in items if item in movieidx2name])

    return movie_recommended_items


def finetuning_evaluate(args, evaluator, epoch, test_gen_dataloader, model, projector, gpt_model, tokenizer_gpt,
                        tokenizer_bert,
                        total_report):
    gpt_model.eval()
    projector.eval()
    model.eval()
    evaluator.log_file.write(f'\n*** test-{epoch} ***\n\n')
    test_cnt = 0
    for batches in tqdm(test_gen_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        if test_cnt == 300:
            break
        else:
            test_cnt += 1
        batch = batches[0]
        with torch.no_grad():
            movie_recommended_items = []
            movie_recommended_items = recommend_top1_item(batch, model)
            if args.conv_pretrained_type == 'none':
                # gen_seqs = gpt_model.generate(**batch['context'], max_new_tokens=args.max_gen_len)
                input_ids = batch['context'].input_ids
                generated = tokenizer_bert.encode('System:', add_special_tokens=False)
                attention_mask = batch['context'].attention_mask
                # past_key_values = None
                cur_len = input_ids.shape[-1]
                unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                gen_len = 0
                while True:
                    if gen_len > args.max_gen_len:
                        break
                    output = gpt_model(input_ids=input_ids.long(), attention_mask=attention_mask,
                                       position_ids=position_ids, conv=True)
                    next_token_logits = output.logits[:, -1, :]
                    # past_key_values = output.past_key_values
                    next_tokens_scores = LogitsProcessorList()(input_ids, next_token_logits)

                    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                    next_tokens = next_tokens * unfinished_sequences + tokenizer_gpt.pad_token_id * (
                            1 - unfinished_sequences)

                    if next_tokens == tokenizer_gpt.vocab['<movie>']:
                        # gen_seq_bert = tokenizer_bert(tokenizer_gpt.batch_decode(input_ids)).input_ids[0]

                        recommended_item_name = movie_recommended_items[0][0]
                        tokenized_name = tokenizer_gpt(recommended_item_name).input_ids
                        tokenized_name = torch.tensor(tokenized_name, device=args.device_id)
                        next_tokens = torch.cat([next_tokens.view(-1), tokenized_name])
                        tokenized_name_len = len(next_tokens)
                        attention_mask = torch.cat(
                            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], tokenized_name_len))],
                            dim=-1)
                        gen_len += tokenized_name_len
                    else:
                        attention_mask = torch.cat(
                            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                            dim=-1)
                        gen_len += 1

                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    input_ids = torch.cat([input_ids, next_tokens.view(input_ids.shape[0], -1)], dim=-1)
                    generated.extend(tokenizer_bert(tokenizer_gpt.decode(next_tokens.view(-1).tolist()),
                                                    add_special_tokens=False).input_ids)
                    unfinished_sequences = unfinished_sequences.mul(
                        (next_tokens[0] != tokenizer_gpt.eos_token_id).long())

                    if unfinished_sequences.max() == 0:
                        break

            else:
                # entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask, user_representation = model.get_representationsWithUser(
                #     batch['context_entities'], batch['context_bert'].input_ids)
                # encoder_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                #                                         entity_padding_mask, user_representation)
                gen_seqs = gpt_model.generate(**batch['context'], prompt_embeds=None,
                                              max_new_tokens=None,
                                              no_repeat_ngram_size=3)
                input_ids = gen_seqs

            # gen_resp_ids = []
            # for gen_seq, length in zip(gen_seqs, batch['context_len']):
            #     gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer_gpt.pad_token_id]
            #     gen_resp_ids.append(gen_seq[length:])
            # evaluator.evaluate(gen_resp_ids, batch['response'], batch['context'], movie_recommended_items, log=True)

            gen_resp_ids2 = []
            for gen_seq, length in zip(input_ids, batch['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer_gpt.pad_token_id]
                gen_resp_ids2.append(gen_seq[length:])
            evaluator.evaluate(gen_resp_ids2, batch['response'], batch['context'], movie_recommended_items, log=True)

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
                       tokenizer_bert,
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
                        tokenizer_bert,
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

            if args.conv_pretrained_type == 'none':
                loss_ft = gpt_model(**batch['context'], conv_labels=batch['response'], conv=True).conv_loss
                loss_pt = gpt_model(**pre_batch['context'], conv_labels=pre_batch['response'], conv=True).conv_loss
            else:
                # with torch.no_grad():
                #     entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask, user_representation = model.get_representationsWithUser(
                #         batch['context_entities'], batch['context_bert'].input_ids)
                #     pre_entity_representations, pre_entity_padding_mask, pre_kg_embedding, pre_token_embedding, pre_token_padding_mask, user_representation = model.get_representationsWithUser(
                #         pre_batch['context_entities'], pre_batch['context_bert'].input_ids)
                #
                # encoder_state, encoder_mask = projector(token_embedding, token_padding_mask, entity_representations,
                #                                         entity_padding_mask, user_representation)
                #
                # pre_encoder_state, pre_encoder_mask = projector(pre_token_embedding, pre_token_padding_mask,
                #                                                 pre_entity_representations,
                #                                                 pre_entity_padding_mask, user_representation)

                loss_ft = gpt_model(**batch['context'], conv_labels=batch['response'], prompt_embeds=None,
                                    conv=True).conv_loss

                loss_pt = gpt_model(**pre_batch['context'], conv_labels=pre_batch['response'], conv=True,
                                    prompt_embeds=None).conv_loss

            loss = loss_ft + ((loss_pt) * args.conv_loss_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.data.float()
        print('Total Loss:\t%.4f' % total_loss)
        print('Loss_pt:\t%.4f\t\t Loss_ft:\t%.4f' % (loss_pt, loss_ft))

        logger.info('[Test]')
        finetuning_evaluate(args, evaluator, epoch + 1, test_gen_dataloader, model, projector, gpt_model, tokenizer_gpt,
                            tokenizer_bert,
                            total_report)
