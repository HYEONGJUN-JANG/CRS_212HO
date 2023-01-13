import re
import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

# year_pattern = re.compile(r'\(\d{4}\)')
slot_pattern = re.compile(r'<movie>')


class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path=None):
        self.tokenizer = tokenizer

        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1, encoding='UTF-8')
            self.log_cnt = 0

    def evaluate(self, preds, labels, contexts, recommended_items, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [
            decoded_pred.replace(self.tokenizer.pad_token, '')
            .replace(self.tokenizer.eos_token, ' ') for decoded_pred in decoded_preds]
            # .replace('<movie>', '').replace('<explain>', '')
            # .replace(decoded_pred[decoded_pred.find('{'):decoded_pred.find('}')+1], '') for decoded_pred in decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace(self.tokenizer.pad_token, '').replace(self.tokenizer.eos_token, ' ') for
                          decoded_label in
                          decoded_labels]

        decoded_labels = [label.strip() for label in decoded_labels]

        decoded_contexts = self.tokenizer.batch_decode(contexts.input_ids, skip_special_tokens=False)
        decoded_contexts = [decoded_context.replace(self.tokenizer.pad_token, '').replace(self.tokenizer.eos_token, ' ')
                            for decoded_context in
                            decoded_contexts]
        decoded_contexts = [context.strip() for context in decoded_contexts]

        if log and hasattr(self, 'log_file'):
            for context, pred, label in zip(decoded_contexts, decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({
                    'context': context,
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_item_ratio(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def evaluate_pretrain(self, titles, response, preds, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                         decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]

        decoded_responses = self.tokenizer.batch_decode(response, skip_special_tokens=False)
        decoded_responses = [decoded_response.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_response in
                             decoded_responses]
        decoded_responses = [response.strip() for response in decoded_responses]

        decoded_titles = self.tokenizer.batch_decode(titles, skip_special_tokens=False)
        decoded_titles = [decoded_title.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_title in
                          decoded_titles]
        decoded_titles = [title.strip() for title in decoded_titles]

        if log and hasattr(self, 'log_file'):
            for response, pred, title in zip(decoded_responses, decoded_preds, decoded_titles):
                self.log_file.write(json.dumps({
                    'pred': pred,
                    'label': title + ' ' + response
                }, ensure_ascii=False) + '\n')

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def compute_item_ratio(self, strs):
        for str in strs:
            # items = re.findall(year_pattern, str)
            # self.metric['item_ratio'] += len(items)
            items = re.findall(slot_pattern, str)
            self.metric['item_ratio'] += len(items)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'dist@1': set(),
            'dist@2': set(),
            'dist@3': set(),
            'dist@4': set(),
            'item_ratio': 0,
        }
        self.sent_cnt = 0
