import random
from abc import ABC
from copy import deepcopy

from loguru import logger
from math import ceil
from tqdm import tqdm
import torch
from utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt


class ReDialDataLoader:
    def __init__(self, dataset, entity_truncate=None, word_truncate=None, padding_idx=0):
        self.dataset = dataset
        self.entity_truncate = entity_truncate
        self.word_truncate = word_truncate
        self.padding_idx = padding_idx

    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
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
        if process_fn is not None:
            dataset = process_fn()
            logger.info('[Finish dataset process before batchify]')

        logger.info(f'[Dataset size: {len(dataset)}]')

        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
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

    def get_rec_data(self, batch_size, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for idx, movie in enumerate(conv_dict['items']):
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_conv_dict['plot'] = conv_dict['plot'][idx]
                    augment_conv_dict['plot_mask'] = conv_dict['plot_mask'][idx]
                    augment_conv_dict['review'] = conv_dict['review'][idx]
                    augment_conv_dict['review_mask'] = conv_dict['review_mask'][idx]
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_context_tokens = []
        batch_plot, batch_plot_mask, batch_review, batch_review_mask = [], [], [], []
        batch_item = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            dialog_history_flatten = sum(conv_dict['context_tokens'], [])
            batch_context_tokens.append(truncate(dialog_history_flatten, self.word_truncate, truncate_tail=False))
            batch_item.append(conv_dict['item'])
            batch_plot.append(conv_dict['plot'])
            batch_plot_mask.append(conv_dict['plot_mask'])
            batch_review.append(conv_dict['review'])
            batch_review_mask.append(conv_dict['review_mask'])


        return (padded_tensor(batch_context_entities, 0, pad_tail=False),
                padded_tensor(batch_context_tokens, 0, pad_tail=False),
                torch.tensor(batch_plot, dtype=torch.long),
                torch.tensor(batch_plot_mask, dtype=torch.long),
                torch.tensor(batch_review, dtype=torch.long),
                torch.tensor(batch_review_mask, dtype=torch.long),
                torch.tensor(batch_item, dtype=torch.long)
                )


    # todo: 아래 retain 뭔지 확인해보기
    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_response, self.pad_token_idx))
