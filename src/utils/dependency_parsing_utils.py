from typing import Dict, Optional
import numpy as np
import torch
from torch import nn
from transformers.file_utils import ModelOutput

"""
Code taken from: https://github.com/Adapter-Hub/adapter-transformers
Original Credits: "How Good is Your Tokenizer? On the
Monolingual Performance of Multilingual Language Models" (Rust et al., 2021) https://arxiv.org/abs/2012.15613
"""


UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        return s


class DependencyParsingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    rel_preds: torch.FloatTensor = None
    arc_preds: torch.FloatTensor = None


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ParsingMetric(Metric):
    """
    based on allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    """

    def __init__(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0

    def add(
        self,
        gold_indices: np.ndarray,
        gold_labels: np.ndarray,
        predicted_indices: np.ndarray,
        predicted_labels: np.ndarray,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        """

        correct_indices = np.equal(predicted_indices, gold_indices)
        correct_labels = np.equal(predicted_labels, gold_labels)
        correct_labels_and_indices = correct_indices * correct_labels

        self._unlabeled_correct += correct_indices.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._total_words += correct_indices.size

    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0


# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        dropout_prob = self.config.get("dropout_prob", model_config.hidden_dropout_prob)
        bias = self.config.get("bias", True)
        for l_id in range(self.config["layers"]):
            if dropout_prob > 0:
                pred_head.append(nn.Dropout(dropout_prob))
            if l_id < self.config["layers"] - 1:
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                if self.config["activation_function"]:
                    pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"], bias=bias))
                elif "num_choices" in self.config:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1, bias=bias))
                else:
                    pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=bias))
                    if self.config["activation_function"]:
                        pred_head.append(Activation_Function_Class(self.config["activation_function"]))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        return None  # override for heads with output embeddings


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)