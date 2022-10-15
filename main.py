import json
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

from dataloader import ReDialDataLoader
from dataset import ContentInformation, ReDialDataset
from model import MovieExpertCRS
from parameters import parse_args
from train import train_recommender
from pretrain import pretrain
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel


## HJ Branch Test
from utils import get_time_kst


def createResultFile(args):
    mdhm = str(
        datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))  # MonthDailyHourMinute .....e.g., 05091040
    results_file_path = f"./results/{mdhm}_train_device_{args.device_id}_name_{args.name}_{args.n_plot}_samples_RLength_{args.max_review_len}_PLength_{args.max_plot_len}_{args.name}.txt"
    if not os.path.exists('./results'): os.mkdir('./results')

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

    # BERT model freeze layers
    if args.n_layer != -1:
        modules = [bert_model.encoder.layer[:args.t_layer - args.n_layer], bert_model.embeddings]  # 2개 남기기
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    content_dataset = ContentInformation(args, REDIAL_DATASET_PATH, tokenizer, args.device_id)
    crs_dataset = ReDialDataset(args, REDIAL_DATASET_PATH, content_dataset, tokenizer)

    train_data = crs_dataset.train_data
    test_data = crs_dataset.test_data

    movie2ids = crs_dataset.movie2id
    num_movie = len(movie2ids)

    # todo: language generation part
    model = MovieExpertCRS(args, bert_model, bert_config.hidden_size, movie2ids, crs_dataset.entity_kg,
                           crs_dataset.n_entity, args.name).to(args.device_id)

    pretrain_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)

    # For pre-training
    if args.name != "none":
        if not args.pretrained:
            # content_data_path = REDIAL_DATASET_PATH + '/content_data.json'

            pretrain(args, model, pretrain_dataloader, pretrained_path)
        else:
            model.load_state_dict(torch.load(pretrained_path))  # state_dict를 불러 온 후, 모델에 저장`

    train_dataloader = ReDialDataLoader(train_data, args.n_sample, args.batch_size, word_truncate=args.max_dialog_len)
    test_dataloader = ReDialDataLoader(test_data, args.n_sample, args.batch_size, word_truncate=args.max_dialog_len)

    content_hit, initial_hit, best_result = train_recommender(args, model, train_dataloader, test_dataloader,
                                                              trained_path, results_file_path,
                                                              pretrain_dataloader)

    return content_hit, initial_hit, best_result
    # todo: result 기록하는 부분 --> train_recommender 안에 구현 완료
    # todo: ???


if __name__ == '__main__':
    args = parse_args()
    main(args)
