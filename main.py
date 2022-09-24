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
def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')


def createResultFile(args):
    mdhm = str(
        datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M'))  # MonthDailyHourMinute .....e.g., 05091040
    results_file_path = f"./results/train_device_{args.device_id}_{mdhm}_name_{args.name}_{args.n_sample}_samples_RLength_{args.max_review_len}_PLength_{args.max_plot_len}_{args.name}.txt"
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
    return results_file_path




if __name__ == '__main__':

    args = parse_args()

    pretrained_path = './saved_model/pretrained_model_' + {args.name} + '.pt'
    trained_path = './saved_model/trained_model+'  + {args.name} + '.pt'

    if torch.cuda.device_count() > 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

    # create result file
    results_file_path = createResultFile(args)

    # Dataset path
    ROOT_PATH = dirname(realpath(__file__))
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    REDIAL_DATASET_PATH = os.path.join(DATA_PATH, 'redial')

    # Load BERT (by using huggingface)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    bert_config = AutoConfig.from_pretrained(args.bert_name)
    bert_model = AutoModel.from_pretrained(args.bert_name, config=bert_config)

    crs_dataset = ReDialDataset(args, REDIAL_DATASET_PATH, tokenizer)
    train_data = crs_dataset.train_data
    test_data = crs_dataset.test_data

    # todo: {crs_id: [entitiy_id, movie_title]} [movie2info.json 으로 한번에 관리하면 편할 듯] --> 완
    movie2ids = crs_dataset.movie2id
    num_movie = len(movie2ids)

    # todo: language generation part
    model = MovieExpertCRS(args, bert_model, bert_config.hidden_size, movie2ids, crs_dataset.entity_kg,
                           crs_dataset.n_entity, args.name).to(args.device_id)

    ## For pre-training
    if args.name != "none":
        if not args.pretrained:
            content_data_path = REDIAL_DATASET_PATH + '/content_data.json'
            content_dataset = ContentInformation(args, content_data_path, tokenizer, args.device_id)

            pretrain_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
            pretrain(args, model, pretrain_dataloader, pretrained_path)
        else:
            model.load_state_dict(torch.load(pretrained_path))  # state_dict를 불러 온 후, 모델에 저장`

    train_dataloader = ReDialDataLoader(train_data, word_truncate=args.max_dialog_len)
    test_dataloader = ReDialDataLoader(test_data, word_truncate=args.max_dialog_len)

    train_recommender(args, model, train_dataloader, test_dataloader, trained_path, results_file_path)

    # todo: result 기록하는 부분 --> train_recommender 안에 구현 완료
    # todo: ???
