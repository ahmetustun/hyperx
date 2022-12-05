"""Implementation of embedding different information sources (task, language, layer_id) for meta adapter layers."""
import json
import torch
import torch.nn as nn
from src.adapters.adapter_utils import linear_layer


def init_x_embeddings(Xs, x_embedding_dim):
    x2embeddings = nn.ParameterDict(dict())
    for x in Xs:
        x_embedding = torch.empty(x_embedding_dim)
        nn.init.normal_(x_embedding)
        x2embeddings[x] = nn.Parameter(x_embedding)
    return x2embeddings


class SourceController(nn.Module):
    def __init__(self, config):
        super(SourceController, self).__init__()
        self.config = config
        self.task_embeddings = init_x_embeddings(config.tasks, config.task_embedding_dim)
        self.lang_embeddings = init_x_embeddings(config.languages, config.language_embedding_dim)

        if config.condition_to_layer_id:
            self.layer_id_embeddings = init_x_embeddings(config.layer_ids, config.layer_id_embedding_dim)
        if config.project_source_embeddings:
            self.source_embedding_MLP = nn.Sequential(
                linear_layer(config.source_embedding_dim, config.source_hidden_dim),
                nn.ReLU(),
                linear_layer(config.source_hidden_dim, config.projected_source_embedding_dim))

    def forward(self, task_name, lang_name, layer_id=None):
        lang_emb = self.lang_embeddings[lang_name]
        source_emb = torch.cat([self.task_embeddings[task_name], lang_emb], dim=0)

        if self.config.condition_to_layer_id:
            source_emb = torch.cat([source_emb, self.layer_id_embeddings[layer_id]])
        if self.config.project_source_embeddings:
            source_emb = self.source_embedding_MLP(source_emb)
        return source_emb