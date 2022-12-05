import numpy as np
import torch
import transformers
from datasets import load_metric

from src.utils.dependency_parsing_utils import ParsingMetric


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def _compute_metrics_token_cls(eval_prediction, label_list):
    metric = load_metric("seqeval")
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
        ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
    }


def _compute_metrics_token_dep(eval_prediction, label_list):
    metric = ParsingMetric()
    predictions, labels = eval_prediction
    rel_labels, arc_labels = labels
    rel_preds, arc_preds = predictions

    mask = np.not_equal(arc_labels, -100)
    predictions_arcs = np.argmax(arc_preds, axis=-1)[mask]

    labels_arcs = arc_labels[mask]

    predictions_rels, labels_rels = rel_preds[mask], rel_labels[mask]
    predictions_rels = predictions_rels[np.arange(len(labels_arcs)), labels_arcs]
    predictions_rels = np.argmax(predictions_rels, axis=-1)

    metric.add(labels_arcs, labels_rels, predictions_arcs, predictions_rels)

    results = metric.get_metric()
    return results


class Eval:
    def __init__(self, datasets_label_list_dict):
        self.label_list_dict = datasets_label_list_dict
        self.task = None

    def set_task(self, task_name):
        self.task = task_name.split(':')[0]

    def get_task(self):
        return self.task

    def compute_metrics(self, eval_prediction):
        label_list = self.label_list_dict[self.task]
        if self.task in ['pos', 'ner']:
            return _compute_metrics_token_cls(eval_prediction, label_list)
        elif self.task in ['dep']:
            return _compute_metrics_token_dep(eval_prediction, label_list)
