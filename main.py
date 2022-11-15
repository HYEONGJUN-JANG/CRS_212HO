import json
import math
from collections import defaultdict
from itertools import product

from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from os.path import dirname, realpath
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pytz import timezone
import sys
import os

from config import gpt2_special_tokens_dict
from dataset_conv import CRSConvDataCollator, CRSConvDataset
from dataloader import ReDialDataLoader
from dataset import ContentInformation, ReDialDataset
from evaluate_conv import ConvEvaluator
from model import MovieExpertCRS
from parameters import parse_args
from train_rec import train_recommender
from pretrain import pretrain
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BartModel, BartTokenizer, AdamW, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM

## HJ Branch Test
from utils import get_time_kst


def createResultFile(args):
    mdhm = str(
        datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))  # MonthDailyHourMinute .....e.g., 05091040
    # results_file_path = f"train_device_{args.device_id}_name_{args.name}_{args.n_plot}_samples_RLength_{args.max_review_len}_PLength_{args.max_plot_len}_{args.name}.txt"
    # if not os.path.exists('./results'): os.mkdir('./results')
    rawSubfolder_name = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d') + '_raw')
    rawFolder_path = os.path.join('./results', rawSubfolder_name)
    if not os.path.exists(rawFolder_path): os.mkdir(rawFolder_path)

    results_file_path = os.path.join(rawFolder_path,
                                     f"{mdhm}_train_device_{args.device_id}_name_{args.name}_{args.n_plot}_samples_RLength_{args.max_review_len}_PLength_{args.max_plot_len}.txt")

    # parameters
    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        result_f.write('Argument List:' + str(sys.argv) + '\n')
        for i, v in vars(args).items():
            result_f.write(f'{i}:{v} || ')
        result_f.write('\n')
        result_f.write('Hit@1\tHit@5\tHit@10\tHit@20\tHit@50\n')
    return results_file_path


def randomize_model(model):
    for module_ in model.named_modules():
        if isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model


def main(args):
    # 22.10.13: path of saved model
    pretrained_path = f'./saved_model/pretrained_model_{args.name}.pt'
    trained_path = f'./saved_model/trained_model_{args.name}.pt'

    # CUDA device check
    # todo: multi-GPU
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

    # create result file
    # todo: tester 와 겹치는 부분 없는지?
    results_file_path = createResultFile(args)

    # Dataset path
    ROOT_PATH = dirname(realpath(__file__))
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    REDIAL_DATASET_PATH = os.path.join(DATA_PATH, 'redial')
    # todo: 삭제??
    content_data_path = REDIAL_DATASET_PATH + '/content_data.json'

    # Load BERT (by using huggingface)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    args.vocab_size = tokenizer.vocab_size
    bert_config = AutoConfig.from_pretrained(args.bert_name)

    if args.t_layer != -1:
        bert_config.num_hidden_layers = args.t_layer
    # bert_config.num_hidden_layers = 1 # 22.09.24 BERT random initialize
    bert_model = AutoModel.from_pretrained(args.bert_name, config=bert_config)
    # bert_model = randomize_model(bert_model) # 22.09.24 BERT random initialize
    # bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # bart = BartModel.from_pretrained('facebook/bart-base')
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = bert_model(**inputs)
    # encoder_vector = bart.encoder(**inputs)

    # BERT model freeze layers
    if args.n_layer != -1:
        if 'bart' in args.bert_name:
            modules = [bert_model.encoder, bert_model.decoder.embed_tokens,
                       bert_model.decoder.layers[:bert_config.num_hidden_layers - args.n_layer]]  # 2개 남기기
        elif 't5' in args.bert_name:
            modules = [bert_model.encoder.block[:bert_config.num_hidden_layers - args.n_layer],
                       bert_model.encoder.embed_tokens]
        else:
            modules = [bert_model.encoder.layer[:bert_config.num_hidden_layers - args.n_layer],
                       bert_model.embeddings]  # 2개 남기기
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    # GPT
    tokenizer_gpt = AutoTokenizer.from_pretrained(args.gpt_name)
    tokenizer_gpt.add_special_tokens(gpt2_special_tokens_dict)
    gpt_model = AutoModelForCausalLM.from_pretrained(args.gpt_name)
    gpt_model.resize_token_embeddings(len(tokenizer_gpt))
    gpt_model.config.pad_token_id = tokenizer.pad_token_id
    gpt_model.config.max_length = 200

    content_dataset = ContentInformation(args, REDIAL_DATASET_PATH, tokenizer, args.device_id)
    crs_dataset = ReDialDataset(args, REDIAL_DATASET_PATH, content_dataset, tokenizer_gpt)

    train_data = crs_dataset.train_data
    valid_data = crs_dataset.valid_data
    test_data = crs_dataset.test_data

    # if args.test:
    #     train_data.extend(valid_data)
    # else:
    #     test_data = valid_data

    movie2ids = crs_dataset.movie2id
    num_movie = len(movie2ids)

    # todo: language generation part
    model = MovieExpertCRS(args, bert_model, gpt_model, bert_config.hidden_size, movie2ids, crs_dataset.entity_kg,
                           crs_dataset.n_entity, args.name).to(args.device_id)

    pretrain_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)

    # For pre-training
    if not args.pretrained:
        # content_data_path = REDIAL_DATASET_PATH + '/content_data.json'
        if 'none' not in args.name:
            pretrain(args, model, pretrain_dataloader, pretrained_path)
    else:
        model.load_state_dict(torch.load(pretrained_path))  # state_dict를 불러 온 후, 모델에 저장`

    if 'rec' in args.task:
        train_rec_dataloader = ReDialDataLoader(train_data, args.n_sample, args.batch_size,
                                                word_truncate=args.max_dialog_len, cls_token=tokenizer.cls_token_id,
                                                task='rec')
        test_rec_dataloader = ReDialDataLoader(test_data, args.n_sample, args.batch_size,
                                               word_truncate=args.max_dialog_len,
                                               cls_token=tokenizer.cls_token_id, task='rec')

        content_hit, initial_hit, best_result = train_recommender(args, model, train_rec_dataloader,
                                                                  test_rec_dataloader,
                                                                  trained_path, results_file_path,
                                                                  pretrain_dataloader)

        return content_hit, initial_hit, best_result
    if 'conv' in args.task:
        # data
        conv_train_dataset = CRSConvDataset(
            REDIAL_DATASET_PATH, 'train', tokenizer_gpt,
            context_max_length=args.context_max_length, resp_max_length=args.max_response_len,
            entity_max_length=args.entity_max_length
        )
        conv_valid_dataset = CRSConvDataset(
            REDIAL_DATASET_PATH, 'valid', tokenizer_gpt,
            context_max_length=args.context_max_length, resp_max_length=args.max_response_len,
            entity_max_length=args.entity_max_length
        )
        conv_test_dataset = CRSConvDataset(
            REDIAL_DATASET_PATH, 'test', tokenizer_gpt,
            context_max_length=args.context_max_length, resp_max_length=args.max_response_len,
            entity_max_length=args.entity_max_length
        )
        # dataloader
        data_collator_teacher = CRSConvDataCollator(
            tokenizer=tokenizer_gpt, device=args.device_id, gen=False,
            context_max_length=args.context_max_length + args.resp_max_length,
            entity_max_length=args.entity_max_length, pad_entity_id=tokenizer_gpt.pad_token_id
        )
        train_dataloader = DataLoader(
            conv_train_dataset,
            batch_size=args.conv_batch_size,
            shuffle=True,
            collate_fn=data_collator_teacher,
        )
        # valid_dataloader = DataLoader(
        #     conv_valid_dataset,
        #     batch_size=args.per_device_eval_batch_size,
        #     num_workers=args.num_workers,
        #     collate_fn=data_collator_teacher,
        # )
        # test_dataloader = DataLoader(
        #     conv_test_dataset,
        #     batch_size=args.per_device_eval_batch_size,
        #     num_workers=args.num_workers,
        #     collate_fn=data_collator_teacher,
        # )
        data_collator_generator = CRSConvDataCollator(
            tokenizer=tokenizer_gpt, device=args.device_id, gen=True,
            context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
            entity_max_length=args.entity_max_length, pad_entity_id=tokenizer_gpt.pad_token_id
        )
        valid_gen_dataloader = DataLoader(
            conv_valid_dataset,
            batch_size=args.conv_batch_size,
            collate_fn=data_collator_generator,
        )
        test_gen_dataloader = DataLoader(
            conv_test_dataset,
            batch_size=args.conv_batch_size,
            collate_fn=data_collator_generator,
        )
        num_update_steps_per_epoch = math.ceil(len(train_dataloader))

        max_train_steps = args.conv_epoch_ft * num_update_steps_per_epoch

        optimizer = AdamW(model.parameters(), lr=args.conv_lr_ft)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, max_train_steps)

        evaluator = ConvEvaluator(tokenizer=tokenizer)
        # TODO: pre-train model load
        total_report = []
        # train loop
        for epoch in range(args.conv_epoch_ft):
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                loss = model.conv_forward(batch['context'], batch['response'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                total_loss += loss.data.float()
            print('Loss:\t%.4f' % total_loss)

            for batch in tqdm(test_gen_dataloader):
                with torch.no_grad():
                    # scores = model.conv_forward(batch['context'], batch['response'])

                    gen_seqs = model.gpt_model.generate(**batch['context'],
                                                        max_new_tokens=args.max_gen_len,
                                                        no_repeat_ngram_size=3)
                    gen_resp_ids = []
                    for gen_seq, length in zip(gen_seqs, batch['context_len']):
                        gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                        gen_resp_ids.append(gen_seq[length:])
                    evaluator.evaluate(gen_resp_ids, batch['response'])
            # metric
            report = evaluator.report()
            test_report = {}
            for k, v in report.items():
                test_report[f'test/{k}'] = v
            # test_loss = np.mean(test_loss)
            # test_report['test/loss'] = test_loss
            test_report['epoch'] = epoch
            logger.info(test_report)
            total_report.append(test_report)
            # if run:
            #     run.log(test_report)
            evaluator.reset_metric()

        return total_report
        # evaluator.log_cnt += 1
        # train_conv_dataloader = ReDialDataLoader(train_data, args.n_sample, args.conv_batch_size,
        #                                          word_truncate=args.max_dialog_len, task='conv')
        # test_conv_dataloader = ReDialDataLoader(test_data, args.n_sample, args.conv_batch_size,
        #                                         word_truncate=args.max_dialog_len, task='conv')

    # todo: result 기록하는 부분 --> train_recommender 안에 구현 완료
    # todo: ???


if __name__ == '__main__':
    args = parse_args()
    main(args)
