import html
import json
import os
import re
from collections import defaultdict
from copy import copy
import random as rand
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
# from config import gpt2_special_tokens_dict
from utils import padded_tensor
import numpy as np
import random

user_template = [
    "User: What movie would you suggest to watch? And tell me about its %s. <|endoftext|>",
    "User: Which movies do you suggest to watch? And tell me about its %s. <|endoftext|>",
    "User: I'm looking for suggestion for good movie with its %s. <|endoftext|>",
    "User: Can you recommend me a movie with its %s? <|endoftext|>"
]

recommend_template = [
    "System: You should watch %s. <explain>",
    "System: I recommend %s. <explain>",
    "System: I suggest %s. <explain>",
    "System: Have you seen %s? <explain>"
]

genre_template = [
    "Its genre is %s.",
    "Its genre is %s.",
    "It is full of %s.",
    "It is %s film."
]

director_template = [
    "It is directed by %s.",
    "%s directed it.",
    "This film is directed by %s.",
    "%s directed this movie."
]

star_template = [
    "It stars %s.",
    "%s acted in this film.",
    "%s is in this movie.",
    "%s appears in this film."
]


class ContentInformationConv(Dataset):
    #
    # Need to bring two datas
    # 1. Content (titles, plots, reviews)
    # 2. Encoder hidden state ( context entities, bert tokenized plots (reviews)
    #

    def __init__(self, args, data_path, tokenizer_gpt, tokenizer_bert):
        super(Dataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_gpt = tokenizer_gpt
        self.data_samples = []
        self.meta_samples = []
        self.device = args.device_id
        self.entity2id = json.load(
            open(os.path.join(data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.movie2id = json.load(open(os.path.join(data_path, 'movie_ids.json'), 'r', encoding='utf-8'))
        self.movie2name = json.load(open(os.path.join(data_path, 'movie2name.json'), 'r', encoding='utf-8'))
        self.read_data()

    def read_data(self):
        logger.info(f'[Conv] content information load')
        for crs_id in tqdm(self.movie2name, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            if self.movie2name[crs_id][0] == -1:
                continue

            meta = eval(self.movie2name[crs_id][2])

            title = "<movie> %s %s" % (self.movie2name[crs_id][1], self.movie2name[crs_id][2])
            meta_input, meta_output = [], []

            # Sampling templates
            idx_user = rand.sample(range(0, len(user_template)), self.args.n_template_sample)
            idx_rec = rand.sample(range(0, len(recommend_template)), self.args.n_template_sample)
            idx_genre = rand.sample(range(0, len(genre_template)), self.args.n_template_sample)
            idx_star = rand.sample(range(0, len(star_template)), self.args.n_template_sample)
            idx_director = rand.sample(range(0, len(director_template)), self.args.n_template_sample)

            # Make synthetic dialog
            user_prompt = [user_template[i] for i in idx_user]
            rec_prompt = [template % title for template in recommend_template]
            rec_prompt = [rec_prompt[i] for i in idx_rec]
            genre_prompt = [template % ', '.join(meta['genre']) for template in genre_template]
            genre_prompt = [genre_prompt[i] for i in idx_genre]
            star_prompt = [template % ', '.join(meta['stars']) for template in star_template]
            star_prompt = [star_prompt[i] for i in idx_star]
            director_prompt = [template % ', '.join(meta['director']) for template in director_template]
            director_prompt = [director_prompt[i] for i in idx_director]

            for prefix in user_prompt:
                for r_prompt in rec_prompt:
                    for g_prompt in genre_prompt:
                        meta_input.append(prefix % 'genre' + r_prompt)
                        meta_output.append(g_prompt)
                    for s_prompt in star_prompt:
                        meta_input.append(prefix % 'star' + r_prompt)
                        meta_output.append(s_prompt)
                    for d_prompt in director_prompt:
                        meta_input.append(prefix % 'director' + r_prompt)
                        meta_output.append(d_prompt)

            tokenzied_meta_input = self.tokenizer_gpt(meta_input, max_length=self.args.max_title_len,
                                                      truncation=True).input_ids
            tokenzied_meta_output = self.tokenizer_gpt(meta_output, max_length=self.args.max_review_len,
                                                       truncation=True).input_ids

            for t_input, t_output in zip(tokenzied_meta_input, tokenzied_meta_output):
                self.meta_samples.append((t_input, t_output))

    def __getitem__(self, idx):
        meta_input = self.meta_samples[idx][0]
        meta_output = self.meta_samples[idx][1]

        return meta_input, meta_output

    def __len__(self):
        return len(self.meta_samples)


class ContentConvCollator:
    def __init__(self, mode, args, tokenizer, tokenizer_bert):
        self.mode = mode
        self.args = args
        self.device = args.device_id
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert

    def __call__(self, data_batch):
        context_batch = defaultdict(list)  # title, content (plot & review)

        resp_batch = []
        context_len_batch = []

        for meta_input, meta_output in data_batch:
            if self.mode == 'train':
                self.tokenizer.padding_side = 'right'
                input_ids = meta_input + meta_output
                input_ids = input_ids[:self.args.max_title_len + self.args.max_gen_len - 1]
                input_ids.append(self.tokenizer.eos_token_id)

                context_batch['input_ids'].append(input_ids)

            elif self.mode == 'test':
                self.tokenizer.padding_side = 'left'
                context_ids = meta_input
                context_len_batch.append(len(context_ids))
                context_batch['input_ids'].append(context_ids)
                resp_batch.append(meta_output)

        input_batch = {}

        context_batch = self.tokenizer.pad(context_batch, padding="max_length",
                                           max_length=self.args.max_title_len + self.args.max_gen_len)
        if self.mode == 'train':
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

        return input_batch


class CRSConvDataset(Dataset):
    #
    # Need to bring three datas
    # 1. Content (titles, plots, reviews)
    # 2. Encoder hidden state ( context entities, bert tokenized plots (reviews)
    # 3. Dialog history
    #
    def __init__(
            self, path, split, tokenizer, tokenizer_bert, content_conv_dataset, debug=False,
    ):
        super(CRSConvDataset, self).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        self.debug = debug
        self.split = split
        self.movie2name = json.load(
            open(os.path.join(path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.movie2id = json.load(
            open(os.path.join(path, 'movie_ids.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.entity2id = json.load(
            open(os.path.join(path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # self.entity_kg = json.load(open(os.path.join(path, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        # self.entity_kg = self._entity_kg_process()

        data_file = os.path.join(path, f'{split}_data.json')
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data_file = json.load(f)

        process_data_file = self._raw_data_process(raw_data_file)
        self.data = process_data_file
        self.content_conv_dataset = content_conv_dataset

    def _raw_data_process(self, raw_data):
        logger.info(f'[Conv] {self.split} - Dataset load')
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
            text = ' '.join(utt['text'])
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

            text_token_ids_bert = self.tokenizer_bert(text_tokens + self.tokenizer_bert.sep_token,
                                                      add_special_tokens=False).input_ids
            text_tokens = text_tokens + self.tokenizer.eos_token
            processed_text_tokens = self.process_utt(text_tokens, self.movie2name,
                                                     replace_movieId=True, remove_movie=True)
            if len(context_tokens) > 0:
                context_tokens[-1] = self.process_utt(context_tokens[-1], self.movie2name,
                                                      replace_movieId=True, remove_movie=True)

                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": self.tokenizer(copy(context_tokens), add_special_tokens=False).input_ids,
                    "context_tokens_bert": copy(context_tokens_bert),
                    "response": self.tokenizer(processed_text_tokens, add_special_tokens=False).input_ids,
                    "context_entities": copy(context_entities)
                }
                if conv['role'] == 'Recommender':
                    augmented_conv_dicts.append(conv_dict)
            context_tokens.append(text_tokens)  # text_tokens
            context_tokens_bert.append(text_token_ids_bert)

            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts

    # def _entity_kg_process(self, SELF_LOOP_ID=185):
    #     edge_list = []  # [(entity, entity, relation)]
    #     for entity in range(self.n_entity):
    #         if str(entity) not in self.entity_kg:
    #             continue
    #         edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
    #         for tail_and_relation in self.entity_kg[str(entity)]:
    #             if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
    #                 edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
    #                 edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))
    #
    #     relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
    #     for h, t, r in edge_list:
    #         relation_cnt[r] += 1
    #     for h, t, r in edge_list:
    #         if relation_cnt[r] > 1000:
    #             if r not in relation2id:
    #                 relation2id[r] = len(relation2id)
    #             edges.add((h, t, relation2id[r]))
    #             entities.add(self.id2entity[h])
    #             entities.add(self.id2entity[t])
    #     return {
    #         'edge': list(edges),
    #         'n_relation': len(relation2id),
    #         'entity': list(entities)
    #     }

    def process_utt(self, utt, movie2name, replace_movieId, remove_movie=False):
        movie_pattern = re.compile(r'@\d+')  # regex

        def convert(match):
            movieid = match.group(0)[1:]
            if movieid in movie2name.keys():
                if remove_movie:
                    return '<movie>'
                movie_name = movie2name[movieid][1]
                return '<movie> ' + movie_name
            else:
                return '<movie>'

        if replace_movieId:
            utt = re.sub(movie_pattern, convert, utt)
        utt = ' '.join(utt.split())
        utt = html.unescape(utt)

        return utt

    def __getitem__(self, item):
        idx = rand.randint(0, len(self.content_conv_dataset) - 1)
        text = self.content_conv_dataset[idx][0]
        title = self.content_conv_dataset[idx][1]

        return text, title, self.data[item]

    def __len__(self):
        return len(self.data)


class CRSConvDataCollator:
    def __init__(
            self, args, device, tokenizer, pad_entity_id, gen=False, use_amp=False, debug=False,
            ignore_pad_token_for_loss=True, context_max_length=None, resp_max_length=None, entity_max_length=None,
            tokenizer_bert=None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug
        self.args = args

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length  # args.context_max_length + args.max_gen_len
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.args.n_meta

        self.pad_entity_id = pad_entity_id

        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __call__(self, data_batch):

        context_batch, pre_context_batch = defaultdict(list), defaultdict(list)
        context_batch_bert, pre_context_batch_bert = defaultdict(list), defaultdict(list)
        entity_batch, pre_entity_batch = [], []
        resp_batch, pre_resp_batch = [], []
        context_len_batch, pre_context_len_batch = [], []

        for meta_input, meta_output, data in data_batch:
            if self.gen:
                self.tokenizer.padding_side = 'left'
                # dialog history
                context_ids = sum(data['context_tokens'], [])
                context_ids = context_ids[-(self.context_max_length - len(self.generate_prompt_ids)):]
                context_len_batch.append(len(context_ids))
                context_ids += self.generate_prompt_ids
                context_batch['input_ids'].append(context_ids)
                resp_batch.append(data['response'])

                # fine-tuning context words
                input_ids_bert = sum(data['context_tokens_bert'], [])
                input_ids_bert = input_ids_bert[-self.args.max_dialog_len + 1:]
                input_ids_bert = [self.tokenizer_bert.cls_token_id] + input_ids_bert
                context_batch_bert['input_ids'].append(input_ids_bert)

                # fine-tuning context entities
                entity_batch.append(data['context_entities'])

                # pre-training
                # todo: content learning 시 eos 붙여줘야 하는거 아닌가?
                pre_context_ids = meta_input
                pre_context_len_batch.append(len(pre_context_ids))
                pre_context_batch['input_ids'].append(pre_context_ids)
                pre_resp_batch.append(meta_output)

            else:
                self.tokenizer.padding_side = 'right'
                # dialog history
                input_ids = sum(data['context_tokens'], []) + data['response']
                input_ids = input_ids[-self.context_max_length:]
                context_batch['input_ids'].append(input_ids)

                # context words
                input_ids_bert = sum(data['context_tokens_bert'], [])
                input_ids_bert = input_ids_bert[-self.args.max_dialog_len + 1:]
                input_ids_bert = [self.tokenizer_bert.cls_token_id] + input_ids_bert
                context_batch_bert['input_ids'].append(input_ids_bert)

                # context entities
                entity_batch.append(data['context_entities'])

                # pre-training
                pre_input_ids = meta_input + meta_output
                pre_input_ids = pre_input_ids[:self.args.max_title_len + self.args.max_gen_len - 1]
                pre_input_ids.append(self.tokenizer.eos_token_id)
                pre_context_batch['input_ids'].append(pre_input_ids)

        input_batch = {}
        pre_input_batch = {}

        # padding
        context_batch = self.tokenizer.pad(
            context_batch, padding="max_length", max_length=self.context_max_length)

        context_batch_bert = self.tokenizer_bert.pad(
            context_batch_bert, padding="max_length", max_length=self.args.max_dialog_len)

        pre_context_batch = self.tokenizer.pad(pre_context_batch, padding="max_length",
                                               max_length=self.args.max_title_len + self.args.max_gen_len)

        if not self.gen:
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            input_batch['response'] = torch.as_tensor(resp_batch, device=self.device)

            pre_resp_batch = pre_context_batch['input_ids']
            pre_resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for
                              resp in pre_resp_batch]
            pre_input_batch['response'] = torch.as_tensor(pre_resp_batch, device=self.device)
        else:
            input_batch['response'] = resp_batch
            input_batch['context_len'] = context_len_batch

            pre_input_batch['response'] = resp_batch
            pre_input_batch['context_len'] = context_len_batch

        # convert to tensor
        # dialog history
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        # fine-tuning: context words
        for k, v in context_batch_bert.items():
            if not isinstance(v, torch.Tensor):
                context_batch_bert[k] = torch.as_tensor(v, device=self.device)
        input_batch['context_bert'] = context_batch_bert

        # fine-tuning: context entities
        entity_batch = padded_tensor(
            entity_batch, max_len=self.entity_max_length
        )
        input_batch['context_entities'] = entity_batch

        # pre-training
        for k, v in pre_context_batch.items():
            if not isinstance(v, torch.Tensor):
                pre_context_batch[k] = torch.as_tensor(v, device=self.device)
        pre_input_batch['context'] = pre_context_batch

        return input_batch, pre_input_batch
