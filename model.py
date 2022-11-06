import torch.nn.functional as F
from torch import nn
import torch
from layers import AdditiveAttention, SelfDotAttention, LastQueryAttention
from torch_geometric.nn import RGCNConv
from transformer import TransformerEncoder
from utils import edge_to_pyg_format


class MovieExpertCRS(nn.Module):
    def __init__(self, args, bert_model, token_emb_dim, movie2ids, entity_kg, n_entity, name):
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

        self.entity_attention = SelfDotAttention(self.kg_emb_dim, self.kg_emb_dim)
        # Dialog
        self.token_emb_dim = token_emb_dim

        if args.word_encoder == 0:
            self.word_encoder = bert_model  # bert or transformer or bart
            if 'bart' in args.bert_name:
                self.word_encoder = bert_model.encoder

        elif args.word_encoder == 1:
            self.token_emb_dim = self.kg_emb_dim
            self.token_embedding = nn.Embedding(self.args.vocab_size, self.token_emb_dim,
                                                self.args.pad_token_id)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.token_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.args.pad_token_id], 0)

            self.word_encoder = TransformerEncoder(
                self.args.n_heads,
                self.args.n_layers,
                self.token_emb_dim,
                self.args.ffn_size,
                self.args.vocab_size,
                self.token_embedding,
                self.args.dropout,
                self.args.attention_dropout,
                self.args.relu_dropout,
                self.args.pad_token_id,
                self.args.learn_positional_embeddings,
                self.args.embeddings_scale,
                self.args.reduction,
                self.args.n_positions
            )

        self.token_attention = AdditiveAttention(self.kg_emb_dim, self.kg_emb_dim)
        self.linear_transformation = nn.Linear(self.token_emb_dim, self.kg_emb_dim)

        # Gating
        self.gating = nn.Linear(2 * self.kg_emb_dim, self.kg_emb_dim)

        # Prediction
        # self.linear_output = nn.Linear(self.token_emb_dim, self.num_movies)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize all parameter (except for pretrained BERT)
        self.initialize()

    # todo: initialize 해줘야 할 parameter check
    def initialize(self):
        # nn.init.xavier_uniform_(self.linear_output.weight)
        nn.init.xavier_uniform_(self.linear_transformation.weight)
        nn.init.xavier_uniform_(self.gating.weight)

        self.entity_attention.initialize()
        self.token_attention.initialize()

    # Input # todo: meta information (entitiy)도 같이 입력
    # plot_token    :   [batch_size, n_plot, max_plot_len]
    # review_token    :   [batch_size, n_plot, max_review_len]
    # plot_meta    :   [batch_size, n_plot, n_meta]
    # target_item   :   [batch_size]
    def pre_forward(self, plot_meta, plot_token, plot_mask, review_meta, review_token, review_mask, target_item,
                    compute_score=False):
        # text = torch.cat([meta_token, plot_token], dim=1)
        # mask = torch.cat([meta_mask, plot_mask], dim=1)
        batch_size = plot_token.shape[0]
        n_plot = plot_token.shape[1]
        max_plot_len = plot_token.shape[2]
        n_review = review_token.shape[1]
        max_review_len = review_token.shape[2]
        n_meta = plot_meta.shape[2]

        if 'plot' in self.name and 'review' in self.name:
            if 'serial' in self.name:  # Cand.3: Review | Plot
                text = torch.cat([plot_token, review_token], dim=1)  # [B, 2N, L]
                mask = torch.cat([plot_mask, review_mask], dim=1)  # [B, 2N, L]
                meta = torch.cat([plot_meta, review_meta], dim=1)  # [B, 2N, L']
                max_len = max_plot_len
                max_meta_len = n_meta
                n_text = n_plot * 2

            else:  # Cand.4: Review & Plot
                plot_token = plot_token.repeat(1, 1, n_plot).view(batch_size, -1, max_plot_len)
                review_token = review_token.repeat(1, n_plot, 1).view(batch_size, -1, max_plot_len)
                text = torch.cat([plot_token, review_token], dim=2)

                plot_mask = plot_mask.repeat(1, 1, n_plot).view(batch_size, -1, max_plot_len)
                review_mask = review_mask.repeat(1, n_plot, 1).view(batch_size, -1, max_plot_len)
                mask = torch.cat([plot_mask, review_mask], dim=2)

                plot_meta = plot_meta.repeat(1, 1, n_plot).view(batch_size, -1, n_meta)
                review_meta = review_meta.repeat(1, n_plot, 1).view(batch_size, -1, n_meta)
                meta = torch.cat([plot_meta, review_meta], dim=2)
                max_len = max_plot_len * 2
                max_meta_len = n_meta * 2
                n_text = n_plot ** 2

                # text = torch.cat([plot_token, review_token], dim=1)
                # mask = torch.cat([plot_mask, review_mask], dim=1)
        elif 'plot' in self.name:  # cand.1: Plot
            text = plot_token
            mask = plot_mask
            meta = plot_meta
            max_len = max_plot_len
            max_meta_len = n_meta
            n_text = n_plot

        elif 'review' in self.name:  # Cand.2: Review
            text = review_token
            mask = review_mask
            meta = review_meta
            max_len = max_plot_len
            max_meta_len = n_meta
            n_text = n_plot
        text = text.to(self.device_id)
        mask = mask.to(self.device_id)

        # [1, B] -> [N, B] -> [N X B]
        target_item = target_item.unsqueeze(1).repeat(1, n_text).view(-1).to(self.device_id)
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

        if self.args.word_encoder == 0:
            text_emb = self.word_encoder(input_ids=text,
                                         attention_mask=mask).last_hidden_state  # [B, L, d] -> [B * N, L, d]
            text_emb = self.linear_transformation(text_emb)  # [B * N, d']
            content_emb = self.token_attention(text_emb, mask=mask)  # [B, d] -> [B * N, d]

        elif self.args.word_encoder == 1:
            text_emb, _ = self.word_encoder(text)  # [B * N , L, d]
            content_emb = self.token_attention(text_emb, mask)  # [B, d] -> [B * N, d]
            # content_emb = self.linear_transformation(content_emb)  # [B * N, d']

        content_emb = self.dropout_pt(content_emb)

        if 'word' in self.args.meta and 'meta' in self.args.meta:
            gate = torch.sigmoid(self.gating(torch.cat([content_emb, entity_attn_rep], dim=1)))
            user_embedding = gate * content_emb + (1 - gate) * entity_attn_rep
        elif 'word' in self.args.meta:
            user_embedding = content_emb
        elif 'meta' in self.args.meta:
            user_embedding = entity_attn_rep

        # content_emb_norm = content_emb / (torch.norm(content_emb, dim=1, keepdim=True) + 1e-10)  # [B, d]
        # entity_attn_rep_norm = entity_attn_rep / (torch.norm(entity_attn_rep, dim=1, keepdim=True) + 1e-10)  # [B, d]
        # affinity = torch.matmul(content_emb_norm, entity_attn_rep_norm.transpose(1, 0))  # [B, B]
        # label = torch.arange(affinity.shape[0]).to(self.device_id)
        # loss_sf = F.cross_entropy(affinity, label)
        # user_embedding = token_attn_rep
        scores = F.linear(user_embedding, kg_embedding)

        loss = self.criterion(scores, target_item)
        if compute_score:
            return scores, target_item
        return loss

    def forward(self, context_entities, context_tokens):

        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)  # (n_entity, entity_dim)
        entity_representations = kg_embedding[context_entities]  # [bs, context_len, entity_dim]
        entity_padding_mask = ~context_entities.eq(self.pad_entity_idx).to(self.device_id)  # (bs, entity_len)
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask,
                                                position=self.args.position)  # (bs, entity_dim)

        token_padding_mask = ~context_tokens.eq(self.pad_entity_idx).to(self.device_id)  # (bs, token_len)
        if self.args.word_encoder == 0:
            token_embedding = self.word_encoder(input_ids=context_tokens.to(self.device_id),
                                                attention_mask=token_padding_mask.to(
                                                    self.device_id)).last_hidden_state  # [bs, token_len, word_dim]
            token_embedding = self.linear_transformation(token_embedding)
            token_attn_rep = self.token_attention(token_embedding, query=entity_attn_rep,
                                                  mask=token_padding_mask)  # [bs, word_dim]

        elif self.args.word_encoder == 1:
            token_embedding, _ = self.word_encoder(context_tokens.to(self.device_id))  # [bs, token_len, word_dim]
            token_attn_rep = self.token_attention(token_embedding, token_padding_mask)  # [bs, word_dim]

        # dropout
        token_attn_rep = self.dropout_ft(token_attn_rep)
        entity_attn_rep = self.dropout_ft(entity_attn_rep)

        # if 'word' in self.args.meta and 'meta' in self.args.meta:
        #     gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        #     user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep
        # elif 'word' in self.args.meta:
        #     user_embedding = token_attn_rep
        # elif 'meta' in self.args.meta:
        #     user_embedding = entity_attn_rep

        gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep

        # user_embedding = token_attn_rep
        scores = F.linear(user_embedding, kg_embedding)
        return scores
