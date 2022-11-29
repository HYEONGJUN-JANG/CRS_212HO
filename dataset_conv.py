import html
import json
import os
import re
from collections import defaultdict
from copy import copy
from random import random

import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# from config import gpt2_special_tokens_dict
from utils import padded_tensor
import numpy as np


class ContentInformationConv(Dataset):
    def __init__(self, args, data_path, tokenizer, device):
        super(Dataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_samples = []
        self.device = device
        self.movie2id = json.load(open('data/redial/movie_ids.json', 'r', encoding='utf-8'))
        self.movie2name = json.load(open('data/redial/movie2name.json', 'r', encoding='utf-8'))
        self.read_data(tokenizer, args.max_plot_len, args.max_review_len)

    def read_data(self, tokenizer, max_plot_len, max_review_len):
        f = open(os.path.join(self.data_path, 'content_data_new.json'), encoding='utf-8')

        data = json.load(f)

        for sample in tqdm(data, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):

            crs_id = sample['crs_id']
            reviews = sample['reviews']
            plots = sample['plots']
            title = "%s (%s)" % (sample['title'], sample['year'])

            if self.movie2name[crs_id][0] == -1:
                continue

            if len(reviews) == 0:
                reviews = ['']
            if len(plots) == 0:
                plots = ['']

            review_prefix = title + self.tokenizer.eos_token
            plot_prefix = title + self.tokenizer.eos_token

            # prefix = title + tokenizer.eos_token
            titled_reviews = [review_prefix + review + tokenizer.eos_token for review in reviews]
            titled_plots = [plot_prefix + plot + tokenizer.eos_token for plot in plots]

            tokenized_title = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(title))
            tokenized_reviews = self.tokenizer(titled_reviews, max_length=max_review_len,
                                               padding='max_length',
                                               truncation=True,
                                               add_special_tokens=True)
            tokenized_plots = self.tokenizer(titled_plots, max_length=max_plot_len,
                                             padding='max_length',
                                             truncation=True,
                                             add_special_tokens=True)

            for i in range(min(len(reviews), self.args.n_review)):
                self.data_samples.append(
                    {"text": tokenized_reviews.input_ids[i], "mask": tokenized_reviews.attention_mask[i]})

            for i in range(min(len(plots), self.args.n_plot)):
                self.data_samples.append(
                    {"text": tokenized_plots.input_ids[i], "mask": tokenized_plots.attention_mask[i]})

        logger.debug('Total number of content samples:\t%d' % len(self.data_samples))

    def __getitem__(self, idx):
        text = self.data_samples[idx]['text']
        mask = self.data_samples[idx]['mask']

        text = torch.LongTensor(text)
        mask = torch.LongTensor(mask)

        return text, mask

    def __len__(self):
        return len(self.data_samples)


class CRSConvDataset(Dataset):
    def __init__(
            self, path, split, tokenizer, tokenizer_bert, debug=False,
            context_max_length=None, resp_max_length=None, entity_max_length=None
    ):
        super(CRSConvDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        # self.prompt_tokenizer = prompt_tokenizer
        self.debug = debug
        self.movie2name = json.load(
            open(os.path.join(path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.moviename2id = {movie_name[1]: movie_id for movie_id, movie_name in self.movie2name.items()}
        self.movie2id = json.load(
            open(os.path.join(path, 'movie_ids.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.entity2id = json.load(
            open(os.path.join(path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(path, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        self.entity_kg = self._entity_kg_process()

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length
        self.resp_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        data_file = os.path.join(path, f'{split}_data.json')
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data_file = json.load(f)
        self.data = []
        process_data_file = self._raw_data_process(raw_data_file)
        self.data = process_data_file
        # self.prepare_data(process_data_file)

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for
                           conversation in tqdm(raw_data,
                                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')]  # 연속해서 나온 대화들 하나로 합침 (예) S1, S2, R1 --> S1 + S2, R1
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            augmented_conv_dicts.extend(self._augment_and_add(conv))  # conversation length 만큼 training sample 생성
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        augmented_convs = []
        last_role = None
        append_role = None

        for utt in dialog:
            # BERT_tokenzier 에 입력하기 위해 @IDX 를 해당 movie의 name으로 replace
            # for idx, word in enumerate(utt['text']):
            #     if word[0] == '@' and word[1:].isnumeric():
            #         utt['text'][idx] = self.movie2name[word[1:]][1]

            text = ' '.join(utt['text'])
            # text_token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if
                         movie in self.entity2id]  # utterance movie(entity2id) 마다 entity2id 저장
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if
                          entity in self.entity2id]  # utterance entity(entity2id) 마다 entity2id 저장

            if utt["role"] == 'Recommender':
                append_role = 'System'
            elif utt["role"] == 'Seeker':
                append_role = 'User'

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += ' ' + text
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": f'{append_role}: {text}',  # role + text
                    "entity": entity_ids,
                    "movie": movie_ids,
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_items = [], [], []
        context_tokens_bert = []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies = conv["text"], conv["entity"], conv["movie"]
            text_tokens = text_tokens + self.tokenizer.eos_token
            # text_token_ids = self.tokenizer(text_tokens, add_special_tokens=False).input_ids
            text_token_ids_bert = self.tokenizer_bert(text_tokens, add_special_tokens=False).input_ids

            plot_meta, plot, plot_mask, review_meta, review, review_mask = [], [], [], [], [], []
            if len(context_tokens) > 0:
                # if len(movies) > 1:
                #     print()
                # for movie in movies:
                # plot_meta.append(self.content_dataset.data_samples[movie]['plot_meta'])
                # plot.append(self.content_dataset.data_samples[movie]['plot'])
                # plot_mask.append(self.content_dataset.data_samples[movie]['plot_mask'])
                # review_meta.append(self.content_dataset.data_samples[movie]['review_meta'])
                # review.append(self.content_dataset.data_samples[movie]['review'])
                # review_mask.append(self.content_dataset.data_samples[movie]['review_mask'])
                mask_text_token = self.process_utt(text_tokens, self.movie2name, replace_movieId=True,
                                                   remove_movie=True)
                context_tokens[-1] = self.process_utt(context_tokens[-1], self.movie2name, replace_movieId=True,
                                                      remove_movie=False)

                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": self.tokenizer(copy(context_tokens), add_special_tokens=False).input_ids,
                    # copy(context_tokens),
                    "context_tokens_bert": copy(context_tokens_bert),
                    "response": self.tokenizer(mask_text_token, add_special_tokens=False).input_ids,  # text_tokens,
                    "context_entities": copy(context_entities)
                    # "context_items": copy(context_items),
                    # "items": movies
                    # "plot_meta": plot_meta,
                    # "plot": plot,
                    # "plot_mask": plot_mask,
                    # "review_meta": review_meta,
                    # "review": review,
                    # "review_mask": review_mask
                }
                if conv['role'] == 'Recommender':
                    augmented_conv_dicts.append(conv_dict)
            context_tokens.append(text_tokens)
            context_tokens_bert.append(text_token_ids_bert)

            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }

    def process_utt(self, utt, movie2name, replace_movieId, remove_movie=False):
        movie_pattern = re.compile(r'@\d+')

        def convert(match):
            movieid = match.group(0)[1:]
            if movieid in movie2name.keys():
                if remove_movie:
                    return '<movie>'
                movie_name = movie2name[movieid][1]
                # movie_name = f'<soi>{movie_name}<eoi>'
                return movie_name
            else:
                return match.group(0)

        if replace_movieId:
            utt = re.sub(movie_pattern, convert, utt)
        utt = ' '.join(utt.split())
        utt = html.unescape(utt)

        return utt

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CRSConvDataCollator:
    def __init__(
            self, tokenizer, device, pad_entity_id, gen=False, use_amp=False, debug=False,
            ignore_pad_token_for_loss=True,
            context_max_length=None, resp_max_length=None, entity_max_length=None,
            tokenizer_bert=None, prompt_max_length=None
    ):
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        # self.prompt_max_length = prompt_max_length
        # if self.prompt_max_length is None:
        #     self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id

        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        context_batch_bert = defaultdict(list)

        # prompt_batch = defaultdict(list)
        entity_batch = []
        resp_batch = []
        context_len_batch = []

        if self.gen:
            self.tokenizer.padding_side = 'left'
            for data in data_batch:
                context_ids = sum(data['context_tokens'], [])
                context_ids = context_ids[-(self.context_max_length - len(self.generate_prompt_ids)):]
                context_len_batch.append(len(context_ids))
                context_ids += self.generate_prompt_ids

                # context_ids += self.generate_prompt_ids
                context_batch['input_ids'].append(context_ids)

                input_ids_bert = sum(data['context_tokens_bert'], [])
                input_ids_bert = input_ids_bert[-self.context_max_length:]
                context_batch_bert['input_ids'].append(input_ids_bert)

                # prompt_batch['input_ids'].append(data['prompt'])
                resp_batch.append(data['response'])
                entity_batch.append(data['context_entities'])
        else:
            self.tokenizer.padding_side = 'right'

            for data in data_batch:
                input_ids = sum(data['context_tokens'], []) + data['response']
                input_ids = input_ids[-self.context_max_length:]
                input_ids_bert = sum(data['context_tokens_bert'], [])
                input_ids_bert = input_ids_bert[-self.context_max_length:]

                context_batch['input_ids'].append(input_ids)
                context_batch_bert['input_ids'].append(input_ids_bert)

                # prompt_batch['input_ids'].append(data['prompt'])
                entity_batch.append(data['context_entities'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length)

        context_batch_bert = self.tokenizer_bert.pad(
            context_batch_bert, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of)

        if not self.gen:
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            input_batch['response'] = torch.as_tensor(resp_batch, device=self.device)
        else:
            input_batch['response'] = resp_batch
            input_batch['context_len'] = context_len_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch
        input_batch['context_bert'] = context_batch_bert

        # prompt_batch = self.prompt_tokenizer.pad(
        #     prompt_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
        #     max_length=self.prompt_max_length
        # )
        # for k, v in prompt_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         prompt_batch[k] = torch.as_tensor(v, device=self.device)
        # input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, max_len=self.entity_max_length
        )
        input_batch['context_entities'] = entity_batch

        return input_batch
