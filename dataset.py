from collections import defaultdict
from copy import copy

from torch.utils.data import Dataset
import torch
import json
from loguru import logger

from tqdm import tqdm
import os

import numpy as np


class ContentInformation(Dataset):
    def __init__(self, args, data_path, tokenizer, movie2id, device):
        super(Dataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_samples = []
        self.device = device
        self.movie2id = movie2id
        self.read_data(tokenizer, args.max_plot_len, args.max_review_len)

    def read_data(self, tokenizer, max_plot_len, max_review_len):

        f = open(self.data_path, encoding='utf-8')
        data = json.load(f)[0]

        # entity2id = json.load(
        #     open(os.path.join('data/redial/', 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        # id2entity = {idx: entity for entity, idx in entity2id.items()}
        # all_entities_name = entity2id.keys()
        for sample in tqdm(data):
            crs_id = sample['crs_id']
            reviews = sample['review']
            plots = sample['summary']
            # title = sample['title']
            # _title = title.replace(' ', '_')

            if crs_id not in self.movie2id:
                continue
            # self.movie2idx[crs_id] = len(self.movie2idx)

            # if 'review' in args.name:
            if reviews is None or len(reviews) == 0:
                continue

            # if 'plot' in args.name:
            if plots is None or len(plots) == 0:
                continue

            tokenzied_reviews = self.tokenizer(reviews, max_length=max_review_len, padding='max_length',
                                               truncation=True,
                                               add_special_tokens=False)

            tokenzied_plots = self.tokenizer(plots, max_length=max_review_len, padding='max_length',
                                             truncation=True,
                                             add_special_tokens=False)

            for i in range(len(reviews)):
                for j in range(len(plots)):
                    review = tokenzied_reviews.input_ids[i]
                    review_mask = tokenzied_reviews.attention_mask[i]
                    plot = tokenzied_plots.input_ids[j]
                    plot_mask = tokenzied_plots.attention_mask[j]

                    self.data_samples.append((crs_id, plot, plot_mask, review, review_mask))

        logger.debug('Total number of content samples:\t%d' % len(self.data_samples))

    def __getitem__(self, item):
        idx, plot, plot_mask, review, review_mask = self.data_samples[item]

        idx = self.movie2id[idx][0]
        """
        Todo: convert crs_id to dbpedia_id. 
        Then, predict original movie id by leveraging its sample of content information. 
        """
        idx = torch.tensor(idx).cuda(self.device)
        plot_token = torch.LongTensor(plot).cuda(self.device)
        plot_mask = torch.LongTensor(plot_mask).cuda(self.device)
        review_token = torch.LongTensor(review).cuda(self.device)
        review_mask = torch.LongTensor(review_mask).cuda(self.device)

        return idx, plot_token, plot_mask, review_token, review_mask

    def __len__(self):
        return len(self.data_samples)


# recommendation mode 와 generation mode에 따라 training sample 이 다르므로, torch.Dataset class 상속 X
class ReDialDataset:
    def __init__(self, args, data_path, tokenizer):
        super(ReDialDataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self._load_other_data()
        self._load_data()

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.data_path, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.data_path, 'train_data.json')}]")
        with open(os.path.join(self.data_path, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.data_path, 'valid_data.json')}]")
        with open(os.path.join(self.data_path, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.data_path, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_other_data(self):
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.data_path, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        # todo: key가 실제 dialog 상 id임. entity_id와 match 필요
        self.movie2name = json.load(
            open(os.path.join(self.data_path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}

        self.movie2id = json.load(
            open(os.path.join(self.data_path, 'movie_ids.json'), 'r', encoding='utf-8'))  # {entity: entity_id}

        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.data_path, 'entity2id.json')} and {os.path.join(self.data_path, 'dbpedia_subkg.json')}]")

        self.entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")

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

    def _load_data(self):
        train_data_raw, valid_data_raw, test_data_raw = self._load_raw_data()  # load raw train, valid, test data

        if self.args.test:
            # For test mode
            train_data_raw.extend(valid_data_raw)
        else:
            # For valid mode
            test_data_raw = valid_data_raw

        self.train_data = self._raw_data_process(train_data_raw)  # training sample 생성
        logger.debug("[Finish train data process]")
        self.test_data = self._raw_data_process(test_data_raw)
        logger.debug("[Finish test data process]")

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for
                           conversation in tqdm(raw_data)]  # 연속해서 나온 대화들 하나로 합침 (예) S1, S2, R1 --> S1 + S2, R1
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))  # conversation length 만큼 training sample 생성
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        augmented_convs = []
        last_role = None

        for utt in dialog:
            # BERT_tokenzier 에 입력하기 위해 @IDX 를 해당 movie의 name으로 replace
            for idx, word in enumerate(utt['text']):
                if word[0] == '@' and word[1:].isnumeric():
                    utt['text'][idx] = self.movie2name[word[1:]][1]  # movie2name -> movie2info 로 변경 필요

            text = ' '.join(utt['text'])
            text_token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if
                         movie in self.entity2id]  # utterance movie(entity2id) 마다 entity2id 저장
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if
                          entity in self.entity2id]  # utterance entity(entity2id) 마다 entity2id 저장

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_items = [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies = conv["text"], conv["entity"], conv["movie"]
            if len(context_tokens) > 0:
                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_items": copy(context_items),
                    "items": movies,
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts
