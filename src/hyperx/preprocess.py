from collections import defaultdict

import torch
import transformers
import numpy as np

# task-type: {0:'mlm', 1:'token-classification', 2:'sequence-classification'}
from src.utils.dependency_parsing_utils import UD_HEAD_LABELS


class Featurizer:

    def __init__(self, tokenizer, max_length=256, label_all_tokens=False):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length if not max_length else max_length
        self.label_all_tokens = label_all_tokens

    def convert_to_features(self, example_batch):
        inputs = list(example_batch["doc"])
        features = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        features["labels"] = example_batch["target"]
        features["task_type"] = [2] * len(inputs)
        return features

    def convert_to_stsb_features(self, example_batch):
        inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
        features = self.tokenizer.batch_encode_plus(
            inputs, max_length=self.max_length, pad_to_max_length=True
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_rte_features(self, example_batch):
        inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
        features = self.tokenizer.batch_encode_plus(
            inputs, max_length=self.max_length, padding='max_length'
        )
        features["labels"] = example_batch["label"]
        return features

    def convert_to_commonsense_qa_features(self, example_batch):
        num_examples = len(example_batch["question"])
        num_choices = len(example_batch["choices"][0]["text"])
        features = {}
        for example_i in range(num_examples):
            choices_inputs = self.tokenizer.batch_encode_plus(
                list(zip(
                    [example_batch["question"][example_i]] * num_choices,
                    example_batch["choices"][example_i]["text"],
                )),
                max_length=self.max_length, pad_to_max_length=True,
            )
            for k, v in choices_inputs.items():
                if k not in features:
                    features[k] = []
                features[k].append(v)
        labels2id = {char: i for i, char in enumerate("ABCDE")}
        # Dummy answers for test
        if example_batch["answerKey"][0]:
            features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
        else:
            features["labels"] = [0] * num_examples
        return features

    def convert_to_mlm_features(self, example_batch):
        # inputs = [line for line in example_batch['text'] if len(line) > 0 and not line.isspace()]
        inputs = [line for line in example_batch['text']]
        features = self.tokenizer.batch_encode_plus(
            inputs, truncation=True, max_length=self.max_length, padding='max_length', return_special_tokens_mask=True
        )
        features["task_type"] = [0] * len(inputs)
        return features

    def _convert_to_token_classification(self, example_batch, tag_name):
        inputs = [line for line in example_batch['tokens']]
        features = self.tokenizer.batch_encode_plus(
            inputs, padding='max_length', truncation=True, max_length=self.max_length, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(example_batch[tag_name]):
            word_ids = features.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if self.label_all_tokens:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        features["labels"] = labels
        # Task type 1 for token classification tasks
        features["task_type"] = [1] * len(inputs)
        return features

    def convert_to_pos_features(self, example_batch):
        return self._convert_to_token_classification(example_batch, 'upos')

    def convert_to_ner_features(self, example_batch):
        return self._convert_to_token_classification(example_batch, 'ner_tags')

    def convert_to_slot_features(self, example_batch):
        return self._convert_to_token_classification(example_batch, 'slots')

    def convert_to_dependency_parsing_features(self, example_batch):
        label_map = {label: i for i, label in enumerate(UD_HEAD_LABELS)}
        features = defaultdict(list)
        for idx, words, heads, deprels in zip(example_batch['idx'], example_batch["tokens"], example_batch["head"], example_batch["deprel"]):
            # clean up -- fixed
            i = 0
            while i < len(heads):
                if heads[i] == "None":
                    del words[i]
                    del heads[i]
                    del deprels[i]
                else:
                    i += 1
            tokens = [self.tokenizer.tokenize(w) for w in words]
            word_lengths = [len(w) for w in tokens]
            tokens_merged = []
            list(map(tokens_merged.extend, tokens))

            if 0 in word_lengths:
                continue
            # Filter out sequences that are too long
            if len(tokens_merged) >= (self.max_length - 2):
                continue

            encoding = self.tokenizer(
                words,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            ignore_index = [-100]

            # pad or truncate arc labels
            labels_arcs = [int(h) for h in heads]
            labels_arcs = labels_arcs + (self.max_length - len(labels_arcs)) * ignore_index

            # convert rel labels from map, pad or truncate if necessary
            labels_rels = [label_map[i.split(":")[0]] for i in deprels]
            labels_rels = labels_rels + (self.max_length - len(labels_rels)) * ignore_index

            # determine start indices of words, pad or truncate if necessary
            word_starts = np.cumsum([1] + word_lengths).tolist()
            word_starts = word_starts + (self.max_length + 1 - len(word_starts)) * ignore_index

            # sanity check lengths
            assert len(input_ids) == self.max_length
            assert len(attention_mask) == self.max_length
            assert len(labels_arcs) == self.max_length
            assert len(labels_rels) == self.max_length
            assert len(word_starts) == self.max_length + 1

            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["word_starts"].append(word_starts)
            features["labels_arcs"].append(labels_arcs)
            features["labels_rels"].append(labels_rels)
            features["task_type"].append([1])

        return dict(features)