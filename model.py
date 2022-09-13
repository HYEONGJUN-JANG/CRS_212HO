import torch.nn.functional as F
from torch import nn
import torch
from layers import AdditiveAttention, SelfDotAttention
from torch_geometric.nn import RGCNConv

from utils import edge_to_pyg_format


class MovieExpertCRS(nn.Module):
    def __init__(self, args, bert_model, token_emb_dim, movie2ids, entity_kg, n_entity, name):
        super(MovieExpertCRS, self).__init__()

        # Setting
        self.args = args
        self.movie2ids = movie2ids # 필요없음
        self.num_movies = len(movie2ids)  # crs_id to dbpedia matching 되면 필요없어짐
        self.name = name  # argument 를 통한 abaltion을 위해 필요
        self.device_id = args.device_id

        # Dialog
        self.token_emb_dim = token_emb_dim
        self.word_encoder = bert_model
        self.token_attention = AdditiveAttention(self.token_emb_dim, self.token_emb_dim)
        self.linear_transformation = nn.Linear(self.token_emb_dim, self.kg_emb_dim)

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

        # Gating
        self.gating = nn.Linear(2 * self.kg_emb_dim, self.kg_emb_dim)

        # Prediction
        self.linear_output = nn.Linear(self.token_emb_dim, self.num_movies)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize all parameter (except for pretrained BERT)
        self.initialize()

    # todo: initialize 해줘야 할 parameter check
    def initialize(self):
        nn.init.xavier_uniform_(self.linear_output.weight)
        nn.init.xavier_uniform_(self.linear_transformation.weight)
        nn.init.xavier_uniform_(self.gating.weight)

        self.entity_attention.initialize()
        self.token_attention.initialize()

    # Input # todo: meta information (entitiy)도 같이 입력
    # plot_token    :   [batch_size, max_plot_len]
    # plot_mask    :   [batch_size, max_plot_len]
    def pretrain(self, plot_token, plot_mask, review_token, review_mask):
        # text = torch.cat([meta_token, plot_token], dim=1)
        # mask = torch.cat([meta_mask, plot_mask], dim=1)
        if 'plot' in self.name and 'review' in self.name:
            text = torch.cat([plot_token, review_token], dim=1)
            mask = torch.cat([plot_mask, review_mask], dim=1)
        elif 'plot' in self.name:
            text = plot_token
            mask = plot_mask
        elif 'review' in self.name:
            text = review_token
            mask = review_mask

        # todo: entitiy 활용해서 pre-train
        # code

        text_emb = self.word_encoder(input_ids=text, attention_mask=mask).last_hidden_state  # [B, L, d]
        content_emb = self.token_attention(text_emb, mask)  # [B, d]

        # todo: MLP layer 로 할 지 dot-prodcut 으로 할 지? (실험)
        scores = self.linear_output(content_emb)  # [B, V]
        # if compute_loss:
        #     return self.criterion(score, label)
        return scores

    def forward(self, context_entities, context_tokens):

        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        entity_representations = kg_embedding[context_entities]  # [B, N, d]
        entity_padding_mask = context_entities.eq(self.pad_entity_idx).to(self.device_id)  # (bs, entity_len)
        entity_attn_rep = self.entity_attention(entity_representations, entity_padding_mask)

        token_padding_mask = context_tokens.eq(self.pad_entity_idx).to(self.device_id)  # (bs, token_len)
        token_embedding = self.word_encoder(input_ids=context_tokens.to(self.device_id),
                                            attention_mask=token_padding_mask.to(
                                                self.device_id)).last_hidden_state  # [B, L, d]
        token_attn_rep = self.token_attention(token_embedding, token_padding_mask)  # [B, d]

        # todo: Linear transformation을 꼭 해줘야 하는지? 해준다면 word 단에서 할 지 sentence 단에서 할 지
        token_attn_rep = self.linear_transformation(token_attn_rep)

        gate = torch.sigmoid(self.gating(torch.cat([token_attn_rep, entity_attn_rep], dim=1)))
        user_embedding = gate * token_attn_rep + (1 - gate) * entity_attn_rep

        scores = F.linear(user_embedding, kg_embedding)
        return scores
