import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import json
from layers import AdditiveAttention, SelfDotAttention, LastQueryAttention
from torch_geometric.nn import RGCNConv
from transformer import TransformerEncoder
from utils import edge_to_pyg_format
from dataclasses import dataclass
from typing import Tuple, Optional
import os
from tqdm import tqdm


@dataclass
class MultiOutput(ModelOutput):
    conv_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class MovieExpertCRS(nn.Module):
    def __init__(self, args, bert_model, bert_config, entity_kg, n_entity, data_path):
        super(MovieExpertCRS, self).__init__()

        # Setting
        self.args = args
        self.device_id = args.device_id
        self.dropout_pt = nn.Dropout(args.dropout_pt)
        self.dropout_ft = nn.Dropout(args.dropout_ft)
        self.movie2ids = json.load(
            open(os.path.join(data_path, 'movie_ids.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        # Entity encoder
        self.n_entity = n_entity
        self.num_bases = args.num_bases
        self.kg_emb_dim = args.kg_emb_dim
        self.n_relation = entity_kg['n_relation']
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(self.device_id)
        self.edge_type = self.edge_type.to(self.device_id)
        self.pad_entity_idx = 0

        # Text encoder
        self.token_emb_dim = bert_config.hidden_size
        self.bert_config = bert_config
        self.word_encoder = bert_model
        self.cls = BertOnlyMLMHead(bert_config)
        self.token_attention = AdditiveAttention(self.kg_emb_dim, self.kg_emb_dim)
        self.linear_transformation = nn.Linear(self.token_emb_dim, self.kg_emb_dim)
        self.entity_proj = nn.Linear(self.kg_emb_dim, self.token_emb_dim)
        self.entity_attention = SelfDotAttention(self.kg_emb_dim, self.kg_emb_dim)

        # Gating
        self.gating = nn.Linear(2 * self.kg_emb_dim, self.kg_emb_dim)
        self.linear_output = nn.Linear(self.kg_emb_dim, self.n_entity)
        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize all parameter (except for pretrained BERT)
        self.initialize()

    # todo: initialize 해줘야 할 parameter check
    def initialize(self):
        nn.init.xavier_uniform_(self.linear_transformation.weight)
        nn.init.xavier_uniform_(self.gating.weight)
        nn.init.xavier_uniform_(self.entity_proj.weight)

        self.entity_attention.initialize()
        self.token_attention.initialize()

    # review_token    :   [batch_size, n_review, max_review_len]
    # review_meta    :   [batch_size, n_review, n_meta]
    # target_item   :   [batch_size]
    def pre_forward(self, review_meta, review_token, review_mask, target_item, compute_score=False):
        n_review = review_token.shape[1]  # number of sampled reviews [N]
        max_review_len = review_token.shape[2]  # length of review text [L]
        n_meta = review_meta.shape[2]  # length of review meta [L']

        text = review_token  # [B, N, L]
        mask = review_mask  # [B, N, L]
        meta = review_meta  # [B, N, L']

        text = text.to(self.device_id)
        mask = mask.to(self.device_id)
        meta = meta.to(self.device_id)

        # [B, 1] -> [N, B] -> [N X B]
        target_item = target_item.unsqueeze(1).repeat(1, n_review).view(-1).to(self.device_id)

        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)

        meta = meta.view(-1, n_meta)  # [B * N, L']
        entity_representations = kg_embedding[meta]  # [B * N, L', d]
        entity_padding_mask = ~meta.eq(self.pad_entity_idx).to(self.device_id)  # (B * N, L')
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask)  # (B *  N, d)
        entity_attn_rep = self.dropout_pt(entity_attn_rep)

        text = text.view(-1, max_review_len)  # [B * N, L]
        mask = mask.view(-1, max_review_len)  # [B * N, L]

        text_emb = self.word_encoder(input_ids=text,
                                     attention_mask=mask).last_hidden_state  # [B * N, L] -> [B * N, L, d]
        proj_text_emb = self.linear_transformation(text_emb)  # [B * N, L, d]
        content_emb = proj_text_emb[:, 0, :]  # [B * N, d]
        content_emb = self.dropout_pt(content_emb)

        gate = torch.sigmoid(self.gating(torch.cat([content_emb, entity_attn_rep], dim=1)))  # [B * N, d * 2]
        user_embedding = gate * content_emb + (1 - gate) * entity_attn_rep  # [B * N, d]

        # if self.args.itemrep == 0:
        scores = F.linear(user_embedding, kg_embedding)  # [B * N, all_entity]
        # else:
        # scores = F.linear(user_embedding, self.kg_encoder.root)  # [B * N, all_entity]
        # scores = self.linear_output(user_embedding)

        loss = self.criterion(scores, target_item)
        if compute_score:
            return scores, target_item
        return loss

    def get_representations(self, context_entities, context_tokens):
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)  # (n_entity, entity_dim)
        entity_padding_mask = ~context_entities.eq(self.pad_entity_idx).to(self.device_id)  # (bs, entity_len)
        token_padding_mask = ~context_tokens.eq(self.pad_entity_idx).to(self.device_id)  # (bs, token_len)

        entity_representations = kg_embedding[context_entities]  # [bs, context_len, entity_dim]

        token_embedding = self.word_encoder(input_ids=context_tokens.to(self.device_id),
                                            attention_mask=token_padding_mask.to(
                                                self.device_id)).last_hidden_state  # [bs, token_len, word_dim]
        return entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask

    def get_representationsWithUser(self, context_entities, context_tokens):
        entity_representations, entity_padding_mask, kg_embedding, token_embedding_prev, token_padding_mask = self.get_representations(
            context_entities,
            context_tokens)

        token_embedding = self.linear_transformation(token_embedding_prev)
        token_attn_rep = token_embedding[:, 0, :]

        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask,
                                                position=self.args.position)  # (bs, entity_dim)

        # dropout
        token_attn_rep = self.dropout_ft(token_attn_rep)
        entity_attn_rep = self.dropout_ft(entity_attn_rep)

        gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep

        return entity_representations, entity_padding_mask, kg_embedding, token_embedding_prev, token_padding_mask, user_embedding

    def make_item_rep(self, movie_id, title, title_mask, review, review_mask):
        if self.args.n_review != 0:
            review = review.view(-1, self.args.max_review_len)  # [B X R, L]
            review_mask = review_mask.view(-1, self.args.max_review_len)  # [B X R, L]
            review_emb = self.word_encoder(input_ids=review, attention_mask=review_mask).last_hidden_state[:, 0,
                         :].view(-1, self.args.n_review, self.token_emb_dim)  # [M X R, L, d]  --> [M, R, d]
            title_emb = self.word_encoder(input_ids=title,
                                  attention_mask=title_mask).last_hidden_state[:, 0, :]  # [M, d]
            # query_embedding = title_emb
            # item_representations = self.item_attention(review_emb, query_embedding, num_review_mask)
            item_representations = (torch.mean(review_emb, dim=1) + title_emb)
        elif self.args.n_review == 0:
            title_emb = self.word_encoder(input_ids=title,
                                  attention_mask=title_mask).last_hidden_state[:, 0, :]  # [M, d]
            item_representations = title_emb

        return item_representations.tolist()

    def forward(self, context_entities, context_tokens, item_review, movie_ids):
        entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask = self.get_representations(
            context_entities,
            context_tokens)

        token_embedding = self.linear_transformation(token_embedding)
        item_review = self.linear_transformation(item_review)
        token_attn_rep = token_embedding[:, 0, :]
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask,
                                                position=self.args.position)  # (bs, entity_dim)

        # dropout
        token_attn_rep = self.dropout_ft(token_attn_rep)
        entity_attn_rep = self.dropout_ft(entity_attn_rep)

        gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep

        # if self.args.itemrep == 0:
        # torch.sum(kg_embedding[:, torch.LongTensor(self.movie2ids)], item_review)
        add_item = torch.zeros(kg_embedding.size(0), kg_embedding.size(1)).to(self.args.device_id)
        movie_ids = movie_ids.tolist()
        for ids in tqdm(movie_ids, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            add_item[ids] = item_review[movie_ids.index(ids)]
        kg_embedding = torch.add(kg_embedding, add_item)
        scores = F.linear(user_embedding, kg_embedding)  # [B * N, all_entity]
        # else:
        #     scores = F.linear(user_embedding, self.kg_encoder.root)  # [B * N, all_entity]
        # scores = self.linear_output(user_embedding)
        return scores
