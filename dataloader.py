import random
from abc import ABC
from copy import deepcopy

from loguru import logger
from math import ceil
from tqdm import tqdm
import torch
from utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt
import numpy as np


class ReDialDataLoader:
    def __init__(self, dataset, n_sample, batch_size, entity_truncate=None, word_truncate=None, padding_idx=0,
                 mode='Test', cls_token=101, task='rec'):
        self.cls_token = cls_token
        self.entity_truncate = entity_truncate
        self.word_truncate = word_truncate
        self.padding_idx = padding_idx
        self.n_sample = n_sample
        self.batch_size = batch_size
        if task == 'rec':
            self.dataset = self.rec_process_fn(dataset, mode)
        elif task == 'conv':
            self.dataset = self.conv_process_fn(dataset)

    def get_data(self, batch_fn, shuffle=True):
        """Collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        dataset = self.dataset
        batch_num = ceil(len(dataset) / self.batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            batch_idx = idx_list[start_idx * self.batch_size: (start_idx + 1) * self.batch_size]
            batch = [dataset[idx] for idx in batch_idx]
            batch = batch_fn(batch)
            if batch == False:
                continue
            else:
                yield (batch)

    def get_conv_data(self, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(self.rec_batchify, shuffle)

    def rec_process_fn(self, dataset, mode):
        augment_dataset = []
        for conv_dict in tqdm(dataset, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            if mode == 'Test':
                if conv_dict['role'] == 'Recommender':
                    for idx, movie in enumerate(conv_dict['items']):
                        augment_conv_dict = deepcopy(conv_dict)
                        augment_conv_dict['item'] = movie
                        augment_conv_dict['plot_meta'] = conv_dict['plot_meta'][idx]
                        augment_conv_dict['plot'] = conv_dict['plot'][idx]
                        augment_conv_dict['plot_mask'] = conv_dict['plot_mask'][idx]
                        augment_conv_dict['review_meta'] = conv_dict['review_meta'][idx]
                        augment_conv_dict['review'] = conv_dict['review'][idx]
                        augment_conv_dict['review_mask'] = conv_dict['review_mask'][idx]
                        augment_dataset.append(augment_conv_dict)
            else:
                for idx, movie in enumerate(conv_dict['items']):
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_conv_dict['plot_meta'] = conv_dict['plot_meta'][idx]
                    augment_conv_dict['plot'] = conv_dict['plot'][idx]
                    augment_conv_dict['plot_mask'] = conv_dict['plot_mask'][idx]
                    augment_conv_dict['review_meta'] = conv_dict['review_meta'][idx]
                    augment_conv_dict['review'] = conv_dict['review'][idx]
                    augment_conv_dict['review_mask'] = conv_dict['review_mask'][idx]
                    augment_dataset.append(augment_conv_dict)

        logger.info('[Finish dataset process before rec batchify]')
        logger.info(f'[Rec Dataset size: {len(augment_dataset)}]')

        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_context_tokens = []
        batch_plot, batch_plot_mask, batch_review, batch_plot_meta, batch_review_meta, batch_review_mask = [], [], [], [], [], []
        batch_item = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            dialog_history_flatten = sum(conv_dict['context_tokens'], [])
            context_tokens = truncate(dialog_history_flatten, self.word_truncate, truncate_tail=False)
            if self.cls_token is not None:
                context_tokens = [self.cls_token] + context_tokens

            batch_context_tokens.append(context_tokens)
            batch_item.append(conv_dict['item'])

            ### Sampling
            plot_exist_num = torch.count_nonzero(torch.sum(torch.tensor(conv_dict['plot_mask']), dim=1))
            review_exist_num = torch.count_nonzero(torch.sum(torch.tensor(conv_dict['review_mask']), dim=1))

            if plot_exist_num == 0 or review_exist_num == 0:
                plot_exist_num = 1
                review_exist_num = 1

            plot_sample_idx = [random.randint(0, plot_exist_num - 1) for _ in range(self.n_sample)]
            review_sample_idx = [random.randint(0, review_exist_num - 1) for _ in range(self.n_sample)]

            batch_plot_meta.append([conv_dict['plot_meta'][k] for k in plot_sample_idx])
            batch_plot.append([conv_dict['plot'][k] for k in plot_sample_idx])
            batch_plot_mask.append([conv_dict['plot_mask'][k] for k in plot_sample_idx])
            batch_review_meta.append([conv_dict['review_meta'][k] for k in review_sample_idx])
            batch_review.append([conv_dict['review'][k] for k in review_sample_idx])
            batch_review_mask.append([conv_dict['review_mask'][k] for k in review_sample_idx])

        return (padded_tensor(batch_context_entities, 0, pad_tail=False),
                padded_tensor(batch_context_tokens, 0, pad_tail=False),
                torch.tensor(batch_plot_meta, dtype=torch.long),
                torch.tensor(batch_plot, dtype=torch.long),
                torch.tensor(batch_plot_mask, dtype=torch.long),
                torch.tensor(batch_review_meta, dtype=torch.long),
                torch.tensor(batch_review, dtype=torch.long),
                torch.tensor(batch_review_mask, dtype=torch.long),
                torch.tensor(batch_item, dtype=torch.long)
                )

    def conv_process_fn(self, dataset):
        augment_dataset = []
        for conv_dict in tqdm(dataset, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            if conv_dict['role'] == 'Recommender':
                augment_dataset.append(conv_dict)
        return augment_dataset

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_response = []
        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_response, self.pad_token_idx))
