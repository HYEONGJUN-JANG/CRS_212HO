import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from layers import AdditiveAttention, SelfDotAttention, LastQueryAttention
from torch_geometric.nn import RGCNConv
from transformer import TransformerEncoder
from utils import edge_to_pyg_format
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MultiOutput(ModelOutput):
    conv_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class Projector(nn.Module):
    def __init__(self, gpt2_config, bert_hidden_size, entity_dim_size, device):
        super(Projector, self).__init__()
        self.gpt_hidden_size = gpt2_config.hidden_size
        self.bert_hidden_size = bert_hidden_size
        self.entity_dim_size = entity_dim_size
        # self.projection_order = projection_order
        self.device = device

        self.token_proj = nn.Sequential(
            nn.Linear(self.bert_hidden_size, self.gpt_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.gpt_hidden_size // 2, self.gpt_hidden_size)
        )

        self.entity_proj = nn.Sequential(
            nn.Linear(self.entity_dim_size, self.gpt_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.gpt_hidden_size // 2, self.gpt_hidden_size)
        )

        self.user_proj = nn.Sequential(
            nn.Linear(self.entity_dim_size, self.gpt_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.gpt_hidden_size // 2, self.gpt_hidden_size)
        )

        self.n_layer = gpt2_config.n_layer
        self.n_block = 2
        self.n_head = gpt2_config.n_head  # head 수는 12
        self.head_dim = self.gpt_hidden_size // self.n_head
        self.prompt_proj2 = nn.Linear(self.gpt_hidden_size, self.n_layer * self.n_block * self.gpt_hidden_size)

    def forward(self, token_emb, token_mask, entity_emb, entity_mask, user_representation):
        token_emb = self.token_proj(token_emb)
        entity_emb = self.entity_proj(entity_emb)
        user_emb = self.user_proj(user_representation)

        encoder_state = torch.cat([token_emb, entity_emb, user_emb.unsqueeze(1)], dim=1)
        encoder_mask = torch.cat([token_mask, entity_mask, torch.ones(token_mask.shape[0], 1, device=self.device)],
                                 dim=1)
        batch_size = encoder_state.shape[0]
        prompt_len = encoder_state.shape[1]
        prompt_embeds = self.prompt_proj2(encoder_state)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds, encoder_mask


class MovieExpertCRS(nn.Module):
    def __init__(self, args, bert_model, bert_config, movie2ids, entity_kg, n_entity, name, n_prefix_rec=10):
        super(MovieExpertCRS, self).__init__()

        # Setting
        self.args = args
        self.movie2ids = movie2ids
        self.name = name  # argument 를 통한 abaltion을 위해 필요
        self.device_id = args.device_id
        self.dropout_pt = nn.Dropout(args.dropout_pt)
        self.dropout_ft = nn.Dropout(args.dropout_ft)
        # R-GCN
        # todo: pre-trainig (R-GCN 자체 or content 내 meta data 를 활용하여?) (후자가 날 듯)
        self.n_entity = n_entity
        self.num_bases = args.num_bases
        self.kg_emb_dim = args.kg_emb_dim
        self.n_relation = entity_kg['n_relation']
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(self.device_id)
        self.edge_type = self.edge_type.to(self.device_id)
        self.pad_entity_idx = 0

        # Dialog
        self.token_emb_dim = bert_config.hidden_size
        self.bert_config = bert_config
        self.word_encoder = bert_model  # bert or transformer or bart
        self.cls = BertOnlyMLMHead(bert_config)
        self.token_attention = AdditiveAttention(self.kg_emb_dim, self.kg_emb_dim)
        self.linear_transformation = nn.Linear(self.token_emb_dim, self.kg_emb_dim)
        self.entity_proj = nn.Linear(self.kg_emb_dim, self.token_emb_dim)
        self.entity_attention = SelfDotAttention(self.kg_emb_dim, self.kg_emb_dim)

        # Gating
        self.gating = nn.Linear(2 * self.kg_emb_dim, self.kg_emb_dim)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize all parameter (except for pretrained BERT)
        self.initialize()

    # todo: initialize 해줘야 할 parameter check
    def initialize(self):
        # nn.init.xavier_uniform_(self.linear_output.weight)
        nn.init.xavier_uniform_(self.linear_transformation.weight)
        nn.init.xavier_uniform_(self.gating.weight)
        nn.init.xavier_uniform_(self.entity_proj.weight)

        self.entity_attention.initialize()
        self.token_attention.initialize()

    # review_token    :   [batch_size, n_review, max_review_len]
    # review_meta    :   [batch_size, n_review, n_meta]
    # target_item   :   [batch_size]
    def pre_forward(self, review_meta, review_token, review_mask, target_item,
                    mask_label,
                    compute_score=False):
        batch_size = review_token.shape[0]
        n_review = review_token.shape[1]
        max_review_len = review_token.shape[2]
        n_meta = review_meta.shape[2]

        text = review_token
        mask = review_mask
        meta = review_meta
        max_len = max_review_len
        max_meta_len = n_meta
        n_text = n_review
        text = text.to(self.device_id)
        mask = mask.to(self.device_id)

        # [B, 1] -> [N, B] -> [N X B]
        target_item = target_item.unsqueeze(1).repeat(1, n_text).view(-1).to(self.device_id)
        # [B, L]
        mask_label = mask_label.repeat(1, n_text).view(-1, max_review_len).to(self.device_id)

        # todo: entitiy 활용해서 pre-train
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)  # (n_entity, entity_dim)

        meta = meta.to(self.device_id)  # [B, N, L']
        meta = meta.view(-1, max_meta_len)  # [B * N, L']
        entity_representations = kg_embedding[meta]  # [B * N, L', d]
        entity_padding_mask = ~meta.eq(self.pad_entity_idx).to(self.device_id)  # (bs, entity_len)
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask)  # (B *  N, d)
        entity_attn_rep = self.dropout_pt(entity_attn_rep)

        # text: [B * N, L]
        text = text.view(-1, max_len)
        mask = mask.view(-1, max_len)

        text_emb = self.word_encoder(input_ids=text,
                                     attention_mask=mask).last_hidden_state  # [B, L, d] -> [B * N, L, d]
        proj_text_emb = self.linear_transformation(text_emb)  # [B * N, d']
        content_emb = proj_text_emb[:, 0, :]
        content_emb = self.dropout_pt(content_emb)

        gate = torch.sigmoid(self.gating(torch.cat([content_emb, entity_attn_rep], dim=1)))
        user_embedding = gate * content_emb + (1 - gate) * entity_attn_rep

        scores = F.linear(user_embedding, kg_embedding)

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

    def forward(self, context_entities, context_tokens):
        entity_representations, entity_padding_mask, kg_embedding, token_embedding, token_padding_mask = self.get_representations(
            context_entities,
            context_tokens)

        token_embedding = self.linear_transformation(token_embedding)
        token_attn_rep = token_embedding[:, 0, :]
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask,
                                                position=self.args.position)  # (bs, entity_dim)

        # dropout
        token_attn_rep = self.dropout_ft(token_attn_rep)
        entity_attn_rep = self.dropout_ft(entity_attn_rep)

        gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep

        scores = F.linear(user_embedding, kg_embedding)
        return scores
