import random
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
    def __init__(self, args, data_path, tokenizer, device):
        super(Dataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        # self.data_samples = []
        self.data_samples = dict()
        self.device = device
        self.entity2id = json.load(
            open(os.path.join(data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.movie2id = json.load(open('data/redial/movie_ids.json', 'r', encoding='utf-8'))
        self.movie2name = json.load(open('data/redial/movie2name.json', 'r', encoding='utf-8'))
        self.read_data(tokenizer, args.max_plot_len, args.max_review_len)
        self.key_list = list(self.data_samples.keys())  # entity id list

    def read_data(self, tokenizer, max_plot_len, max_review_len):
        f = open(os.path.join(self.data_path, 'content_data_new.json'), encoding='utf-8')

        data = json.load(f)

        # entity2id = json.load(
        #     open(os.path.join('data/redial/', 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        # id2entity = {idx: entity for entity, idx in entity2id.items()}
        # all_entities_name = entity2id.keys()
        for sample in tqdm(data, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            review_list, plot_list = [], []
            review_mask_list, plot_mask_list, reviews_meta_list, plots_meta_list = [], [], [], []

            crs_id = sample['crs_id']
            reviews = sample['reviews']
            plots = sample['plots']
            plots_meta = sample['plots_meta']
            reviews_meta = sample['reviews_meta']
            title = "%s (%s)" % (sample['title'], sample['year'])
            # title = sample['title']
            # _title = title.replace(' ', '_')

            if self.movie2name[crs_id][0] == -1:
                continue

            if len(reviews) == 0:
                reviews = ['']
                reviews_meta = [[]]
            # if 'plot' in args.name:
            if len(plots) == 0:
                plots = ['']
                plots_meta = [[]]

            # Filter out movie name in plots, reviews
            # reviews = [review.replace(sample['title'], self.tokenizer.mask_token) for review in reviews]
            # plots = [plot.replace(sample['title'], self.tokenizer.mask_token) for plot in plots]

            # prefix = title + tokenizer.sep_token
            # masked_title =
            # tokenized_title = self.tokenizer(title, add_special_tokens=False).input_ids
            # masked_prefix = self.tokenizer.mask_token * len(tokenized_title) + self.tokenizer.sep_token
            # masked_review_prefix = "The review of " + self.tokenizer.mask_token * len(
            #     tokenized_title) + self.tokenizer.sep_token
            # masked_plot_prefix = "The plot of " + self.tokenizer.mask_token * len(
            #     tokenized_title) + self.tokenizer.sep_token

            # prefixed_reviews = [masked_prefix + review for review in reviews]
            # prefixed_plots = [masked_prefix + plot for plot in plots]
            mask_label = [-100] * max_review_len
            # mask_label[1:1 + len(tokenized_title)] = tokenized_title

            tokenized_reviews = self.tokenizer(reviews, max_length=max_review_len,
                                               padding='max_length',
                                               truncation=True,
                                               add_special_tokens=True)
            tokenized_plots = self.tokenizer(plots, max_length=max_plot_len,
                                             padding='max_length',
                                             truncation=True,
                                             add_special_tokens=True)

            if self.args.word_encoder == 2:
                review_lens = [sum(mask) for mask in tokenized_reviews.attention_mask]
                for tokenized_review, last_idx in zip(tokenized_reviews.input_ids, review_lens):
                    tokenized_review[last_idx - 1] = tokenizer.cls_token_id

                plot_lens = [sum(mask) for mask in tokenized_plots.attention_mask]
                for tokenized_plot, last_idx in zip(tokenized_plots.input_ids, plot_lens):
                    tokenized_plot[last_idx - 1] = tokenizer.cls_token_id


            for idx, meta in enumerate(reviews_meta):
                reviews_meta[idx] = [self.entity2id[entity] for entity in meta][:self.args.n_meta]
                reviews_meta[idx] = reviews_meta[idx] + [0] * (self.args.n_meta - len(meta))

            for idx, meta in enumerate(plots_meta):
                plots_meta[idx] = [self.entity2id[entity] for entity in meta][:self.args.n_meta]
                plots_meta[idx] = plots_meta[idx] + [0] * (self.args.n_meta - len(meta))

            for i in range(min(len(reviews), self.args.n_review)):
                review_list.append(tokenized_reviews.input_ids[i])
                review_mask_list.append(tokenized_reviews.attention_mask[i])
                reviews_meta_list.append(reviews_meta[i])

            for i in range(self.args.n_review - len(reviews)):
                zero_vector = [0] * max_review_len
                review_list.append(zero_vector)
                review_mask_list.append(zero_vector)

            for i in range(min(len(plots), self.args.n_plot)):
                plot_list.append(tokenized_plots.input_ids[i])
                plot_mask_list.append(tokenized_plots.attention_mask[i])
                plots_meta_list.append(plots_meta[i])

            for i in range(self.args.n_plot - len(plots)):
                zero_vector = [0] * max_plot_len
                plot_list.append(zero_vector)
                plot_mask_list.append(zero_vector)

            self.data_samples[self.movie2name[crs_id][0]] = {"plot": plot_list, "plot_mask": plot_mask_list,
                                                             "review": review_list,
                                                             "review_mask": review_mask_list,
                                                             "review_meta": reviews_meta_list,
                                                             "plot_meta": plots_meta_list,
                                                             "mask_label": mask_label
                                                             }

        logger.debug('Total number of content samples:\t%d' % len(self.data_samples))

    def __getitem__(self, item):
        idx = self.key_list[item]  # dbpedia id
        plot_token = self.data_samples[idx]['plot']
        plot_mask = self.data_samples[idx]['plot_mask']
        review_token = self.data_samples[idx]['review']
        review_mask = self.data_samples[idx]['review_mask']
        review_meta = self.data_samples[idx]['review_meta']
        plot_meta = self.data_samples[idx]['plot_meta']
        mask_label = self.data_samples[idx]['mask_label']
        # ### Sampling
        # if len(meta) > 0:
        #     sample_idx = [random.randint(0, len(meta) - 1) for _ in range(self.args.n_sample)]
        #     entities = [meta[k] for k in sample_idx]
        # else:
        #     entities = [0] * self.args.n_sample

        plot_exist_num = np.count_nonzero(np.sum(np.array(plot_mask), axis=1))
        review_exist_num = np.count_nonzero(np.sum(np.array(review_mask), axis=1))

        # 221013. 기존코드는 plot 혹은 review가 0이라면, 둘 다 1로 설정; plot 은 0인데 , review 는 10개라면?
        if plot_exist_num == 0:
            plot_exist_num = 1
        if review_exist_num == 0:
            review_exist_num = 1

        plot_sample_idx = [random.randint(0, plot_exist_num - 1) for _ in range(self.args.n_sample)]
        review_sample_idx = [random.randint(0, review_exist_num - 1) for _ in range(self.args.n_sample)]

        plot_token = [plot_token[k] for k in plot_sample_idx]
        plot_mask = [plot_mask[k] for k in plot_sample_idx]
        plot_meta = [plot_meta[k] for k in plot_sample_idx]

        review_token = [review_token[k] for k in review_sample_idx]
        review_mask = [review_mask[k] for k in review_sample_idx]
        review_meta = [review_meta[k] for k in review_sample_idx]

        idx = torch.tensor(idx)
        plot_token = torch.LongTensor(plot_token)
        plot_mask = torch.LongTensor(plot_mask)
        plot_meta = torch.LongTensor(plot_meta)
        review_token = torch.LongTensor(review_token)
        review_mask = torch.LongTensor(review_mask)
        review_meta = torch.LongTensor(review_meta)
        mask_label = torch.LongTensor(mask_label)

        return idx, plot_meta, plot_token, plot_mask, review_meta, review_token, review_mask, mask_label

    def __len__(self):
        return len(self.data_samples)


# recommendation mode 와 generation model에 따라 training sample 이 다르므로, torch.Dataset class 상속 X
class ReDialDataset:
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

    def __init__(self, args, data_path, content_dataset, tokenizer, sep_token):
        super(ReDialDataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.content_dataset = content_dataset
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self._load_other_data()
        self._load_data()

    def _load_other_data(self):
        # todo: KG information 분리 시켜야 함
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.data_path, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        self.entity_kg = self._entity_kg_process()

        self.movie2name = json.load(
            open(os.path.join(self.data_path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}

        self.movie2id = json.load(
            open(os.path.join(self.data_path, 'movie_ids.json'), 'r', encoding='utf-8'))  # {entity: entity_id}

        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.data_path, 'entity2id.json')} and {os.path.join(self.data_path, 'dbpedia_subkg.json')}]")

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

        self.train_data = self._raw_data_process(train_data_raw)  # training sample 생성
        logger.debug("[Finish train data process]")
        self.test_data = self._raw_data_process(test_data_raw)
        logger.debug("[Finish test data process]")
        self.valid_data = self._raw_data_process(valid_data_raw)
        logger.debug("[Finish valid data process]")

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for
                           conversation in tqdm(raw_data,
                                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')]  # 연속해서 나온 대화들 하나로 합침 (예) S1, S2, R1 --> S1 + S2, R1
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
                    utt['text'][idx] = self.movie2name[word[1:]][1]

            text = ' '.join(utt['text'])
            # text_token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if
                         movie in self.entity2id]  # utterance movie(entity2id) 마다 entity2id 저장
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if
                          entity in self.entity2id]  # utterance entity(entity2id) 마다 entity2id 저장

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += ' ' + text
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:

                if utt["role"] == 'Recommender':
                    role_name = 'System'
                else:
                    role_name = 'User'

                augmented_convs.append({
                    "role": utt["role"],
                    "text": f'{role_name}: {text}',  # role + text
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
            text_tokens = text_tokens + self.sep_token
            text_token_ids = self.tokenizer(text_tokens, add_special_tokens=False).input_ids
            plot_meta, plot, plot_mask, review_meta, review, review_mask, mask_label = [], [], [], [], [], [], []
            if len(context_tokens) > 0:
                # if len(movies) > 1:
                #     print()
                for movie in movies:
                    plot_meta.append(self.content_dataset.data_samples[movie]['plot_meta'])
                    plot.append(self.content_dataset.data_samples[movie]['plot'])
                    plot_mask.append(self.content_dataset.data_samples[movie]['plot_mask'])
                    review_meta.append(self.content_dataset.data_samples[movie]['review_meta'])
                    review.append(self.content_dataset.data_samples[movie]['review'])
                    review_mask.append(self.content_dataset.data_samples[movie]['review_mask'])
                    mask_label.append(self.content_dataset.data_samples[movie]['mask_label'])

                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_token_ids,  # text_tokens,
                    "context_entities": copy(context_entities),
                    "context_items": copy(context_items),
                    "items": movies,
                    "plot_meta": plot_meta,
                    "plot": plot,
                    "plot_mask": plot_mask,
                    "review_meta": review_meta,
                    "review": review,
                    "review_mask": review_mask,
                    "mask_label": mask_label

                }
                augmented_conv_dicts.append(conv_dict)
            context_tokens.append(text_token_ids)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts
