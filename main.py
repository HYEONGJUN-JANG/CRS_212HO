import json
import math
from collections import defaultdict
from itertools import product

from loguru import logger
from torch.nn import CrossEntropyLoss
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

from dataset_kg import KGInformation
from train_conv import train_conversation
from config import gpt2_special_tokens_dict, bert_special_tokens_dict
from dataset_conv import CRSConvDataCollator, CRSConvDataset, ContentInformationConv, ContentConvCollator
from dataloader import CRSDataLoader
from dataset_rec import ContentInformation, CRSDatasetRec
from evaluate_conv import ConvEvaluator
from model import MovieExpertCRS
from model_gpt2 import PromptGPT2forCRS
from parameters import parse_args
from pretrain_conv import pretrain_conv
from train_rec import train_recommender
from pretrain_rec import pretrain

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BartModel, BartTokenizer, AdamW, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM, GPT2Model, GPT2LMHeadModel

from utils import get_time_kst


def createResultFile(args):
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    rawSubfolder_name = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d') + '_raw')
    rawFolder_path = os.path.join('./results', rawSubfolder_name)
    if not os.path.exists(rawFolder_path): os.mkdir(rawFolder_path)

    if 'rec' in args.task:
        results_file_path = os.path.join(rawFolder_path,
                                         f"[REC]{mdhm}_train_device_{args.device_id}_name_{args.name}_{args.n_review}_samples_RLength_{args.max_review_len}.txt")

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

    elif 'conv' in args.task:
        conv_result_file_path = os.path.join(rawFolder_path,
                                             f"[CONV]{mdhm}_train_device_{args.device_id}_name_{args.name}.txt")
        pre_conv_result_file_path = os.path.join(rawFolder_path,
                                                 f"[PRECONV]{mdhm}_train_device_{args.device_id}_name_{args.name}.txt")

        with open(conv_result_file_path, 'a', encoding='utf-8') as result_f:
            result_f.write(
                '\n=================================================\n')
            result_f.write(get_time_kst())
            result_f.write('\n')
            result_f.write('Argument List:' + str(sys.argv) + '\n')
            for i, v in vars(args).items():
                result_f.write(f'{i}:{v} || ')
            result_f.write('\n')

        with open(pre_conv_result_file_path, 'a', encoding='utf-8') as result_f:
            result_f.write(
                '\n=================================================\n')
            result_f.write(get_time_kst())
            result_f.write('\n')
            result_f.write('Argument List:' + str(sys.argv) + '\n')
            for i, v in vars(args).items():
                result_f.write(f'{i}:{v} || ')
            result_f.write('\n')

        return conv_result_file_path, pre_conv_result_file_path


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
    if 'redial' in args.dataset_path:
        pretrained_path = f'./saved_model/{args.task}/redial/pretrained_model_{args.name}.pt'
        trained_path = f'./saved_model/{args.task}/redial/trained_model_{args.name}.pt'
        best_rec_path = f'./saved_model/rec/redial/trained_model_best.pt'
        best_rec_pretrained_path = f'./saved_model/rec/redial/pretrained_model_best.pt'
        best_conv_pretrained_path = f'./saved_model/conv/redial/pretrained_model_best.pt'
    elif 'inspired' in args.dataset_path:
        pretrained_path = f'./saved_model/{args.task}/inspired/pretrained_model_{args.name}.pt'
        trained_path = f'./saved_model/{args.task}/inspired/trained_model_{args.name}.pt'
        best_rec_path = f'./saved_model/rec/inspired/trained_model_best.pt'
        best_rec_pretrained_path = f'./saved_model/rec/redial/pretrained_model_best.pt'
        best_conv_pretrained_path = f'./saved_model/conv/inspired/pretrained_model_best.pt'

    # Dataset path
    ROOT_PATH = dirname(realpath(__file__))
    DATASET_PATH = os.path.join(ROOT_PATH, args.dataset_path)

    # Load BERT (by using huggingface)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)
    bert_config = AutoConfig.from_pretrained(args.bert_name)
    args.vocab_size = tokenizer.vocab_size
    if args.t_layer != -1:
        bert_config.num_hidden_layers = args.t_layer
    if 'gpt' in args.bert_name.lower():
        bert_model = AutoModel.from_pretrained(args.gpt_name)
    else:
        bert_model = AutoModel.from_pretrained(args.bert_name)
    bert_model.resize_token_embeddings(len(tokenizer))

    # BERT model freeze layers
    if args.n_layer != -1:
        modules = [bert_model.encoder.layer[:bert_config.num_hidden_layers - args.n_layer],
                   bert_model.embeddings]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    # GPT
    tokenizer_gpt = AutoTokenizer.from_pretrained(args.gpt_name)
    tokenizer_gpt.add_special_tokens(gpt2_special_tokens_dict)
    gpt_config = AutoConfig.from_pretrained(args.gpt_name)

    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_name, config=gpt_config)
    gpt_model.resize_token_embeddings(len(tokenizer_gpt))
    gpt_model.config.pad_token_id = tokenizer_gpt.pad_token_id
    gpt_model = gpt_model.to(args.device_id)

    # GPT model freeze layers
    if args.gpt_n_layer != -1:
        if 'gpt' in args.gpt_name:
            modules = [gpt_model.h[:gpt_config.num_hidden_layers - args.n_layer],
                       gpt_model.wte, gpt_model.wpe]  # 2개 남기기
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    kg_information = KGInformation(args, DATASET_PATH)

    # Load expert model
    model = MovieExpertCRS(args, bert_model, bert_config, kg_information.entity_kg, kg_information.n_entity,
                           DATASET_PATH).to(args.device_id)

    if 'rec' in args.task:
        # create result file
        results_file_path = createResultFile(args)
        content_dataset = ContentInformation(args, DATASET_PATH, tokenizer, args.device_id)
        crs_dataset = CRSDatasetRec(args, DATASET_PATH, content_dataset, tokenizer, kg_information)
        train_data = crs_dataset.train_data
        valid_data = crs_dataset.valid_data
        test_data = crs_dataset.test_data

        pretrain_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)

        # For pre-training
        if not args.pretrained:
            pretrain(args, model, pretrain_dataloader, pretrained_path)
        else:
            model.load_state_dict(torch.load(best_rec_pretrained_path))  # state_dict를 불러 온 후, 모델에 저장`

        type = 'bert'
        if args.dataset_path == 'data/inspired':
            for param in model.word_encoder.parameters():
                param.requires_grad = False

        train_rec_dataloader = CRSDataLoader(train_data, args.n_sample, args.batch_size,
                                             word_truncate=args.max_dialog_len, cls_token=tokenizer.cls_token_id,
                                             task='rec', type=type)
        valid_rec_dataloader = CRSDataLoader(valid_data, args.n_sample, args.batch_size,
                                             word_truncate=args.max_dialog_len,
                                             cls_token=tokenizer.cls_token_id, task='rec', type=type)
        test_rec_dataloader = CRSDataLoader(test_data, args.n_sample, args.batch_size,
                                            word_truncate=args.max_dialog_len,
                                            cls_token=tokenizer.cls_token_id, task='rec', type=type)

        if args.mode == 'test':
            content_hit, initial_hit, best_result = train_recommender(args, model, train_rec_dataloader,
                                                                      test_rec_dataloader,
                                                                      trained_path, results_file_path,
                                                                      pretrain_dataloader)
        elif args.mode == 'valid':
            content_hit, initial_hit, best_result = train_recommender(args, model, train_rec_dataloader,
                                                                      valid_rec_dataloader,
                                                                      trained_path, results_file_path,
                                                                      pretrain_dataloader)

        return content_hit, initial_hit, best_result

    if 'conv' in args.task:
        # Create Result files
        conv_results_file_path, pre_conv_result_file_path = createResultFile(args)
        # Load fine-tuned recommender model
        if os.path.isfile(best_rec_path):
            logger.info(f'Load pretrained file\t{best_rec_path}')
            model.load_state_dict(torch.load(best_rec_path, map_location='cuda:%d' % args.device_id))
        for param in model.parameters():
            param.requires_grad = False

        # Synthetic dataset
        content_conv_dataset = ContentInformationConv(args, DATASET_PATH, tokenizer_gpt, tokenizer)
        content_conv_train_collator = ContentConvCollator('train', args, tokenizer_gpt, tokenizer)
        content_conv_test_collator = ContentConvCollator('test', args, tokenizer_gpt, tokenizer)
        pretrain_conv_dataloader_train = DataLoader(content_conv_dataset, batch_size=args.conv_batch_size,
                                                    shuffle=True, collate_fn=content_conv_train_collator)
        pretrain_conv_dataloader_test = DataLoader(content_conv_dataset, batch_size=args.conv_pre_eval_batch_size,
                                                   shuffle=False, collate_fn=content_conv_test_collator)
        if not args.conv_pretrained:
            # Pre-train
            pretrain_conv(args, model, gpt_model, gpt_config, tokenizer_gpt, pretrain_conv_dataloader_train,
                          pretrain_dataloader_test=pretrain_conv_dataloader_test,
                          path=pre_conv_result_file_path, save_path=pretrained_path)
        else:
            # Load saved model for pre-training
            gpt_model.load_state_dict(torch.load(best_conv_pretrained_path,
                                                 map_location='cuda:%d' % args.device_id))
            logger.info(f'Load pretrained conv file\t{best_conv_pretrained_path}')

        # Dialog history
        conv_train_dataset = CRSConvDataset(
            DATASET_PATH, 'train', tokenizer_gpt, tokenizer, content_conv_dataset,
        )
        conv_valid_dataset = CRSConvDataset(
            DATASET_PATH, 'valid', tokenizer_gpt, tokenizer, content_conv_dataset,
        )
        conv_test_dataset = CRSConvDataset(
            DATASET_PATH, 'test', tokenizer_gpt, tokenizer, content_conv_dataset,
        )
        # dataloader
        data_collator_teacher = CRSConvDataCollator(
            args, tokenizer=tokenizer_gpt, tokenizer_bert=tokenizer, device=args.device_id, gen=False,
            context_max_length=args.context_max_length + args.max_gen_len,
            entity_max_length=args.entity_max_length, pad_entity_id=tokenizer_gpt.pad_token_id
        )
        train_dataloader = DataLoader(
            conv_train_dataset,
            batch_size=args.conv_batch_size,
            shuffle=True,
            collate_fn=data_collator_teacher,
        )

        data_collator_generator = CRSConvDataCollator(
            args, tokenizer=tokenizer_gpt, tokenizer_bert=tokenizer, device=args.device_id, gen=True,
            context_max_length=args.context_max_length, resp_max_length=args.max_gen_len,
            entity_max_length=args.entity_max_length, pad_entity_id=tokenizer_gpt.pad_token_id
        )
        valid_gen_dataloader = DataLoader(
            conv_valid_dataset,
            batch_size=args.conv_batch_size,
            collate_fn=data_collator_generator,
        )
        test_gen_dataloader = DataLoader(
            conv_test_dataset,
            batch_size=args.gen_batch_size,
            collate_fn=data_collator_generator,
        )
        # train & test
        if args.mode == 'test':
            train_conversation(args, model, train_dataloader, test_gen_dataloader, pretrain_conv_dataloader_test,
                               gpt_model, gpt_config, tokenizer_gpt, tokenizer, conv_results_file_path)
        elif args.mode == 'valid':
            train_conversation(args, model, train_dataloader, valid_gen_dataloader, pretrain_conv_dataloader_test,
                               gpt_model, gpt_config, tokenizer_gpt, tokenizer, conv_results_file_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
