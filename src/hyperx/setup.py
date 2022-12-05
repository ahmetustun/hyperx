import os

import transformers
from datasets import load_dataset
import src.hg_utils.bert.modeling_bert as hpx_base_models
import src.hg_utils.bert.configuration_bert as hpx_base_config
from src.hyperx.datacollator import DataCollator
from src.hyperx.eval import Eval
from src.hyperx.model import Hyperx
from src.hyperx.preprocess import Featurizer
from src.utils.dependency_parsing_utils import UD_HEAD_LABELS


class HyperxSetup:
    def __init__(self, args):
        self.args = args
        self.tasks = args.tasks
        self.languages = args.languages
        self.model_name = args.model_name_or_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, use_fast=True,
                                                                    add_prefix_space=True)
        featurizer = Featurizer(tokenizer=self.tokenizer, max_length=args.max_length)

        self.data_collator = DataCollator(self.tokenizer)

        self.task_model_dict = {
            "ner": hpx_base_models.BertForTokenClassification,
            "pos": hpx_base_models.BertForTokenClassification,
            "mlm": hpx_base_models.BertForMaskedLM,
            "dep": hpx_base_models.BertForDependencyParsing,
        }

        self.task_conf_dict = {
            "ner": hpx_base_config.BertConfig.from_pretrained(self.model_name, num_labels=7),
            "pos": hpx_base_config.BertConfig.from_pretrained(self.model_name, num_labels=18),
            "mlm": hpx_base_config.BertConfig.from_pretrained(self.model_name),
            "dep": hpx_base_config.BertConfig.from_pretrained(self.model_name, num_labels=38),
        }

        self.convert_func_dict = {
            "ner": featurizer.convert_to_ner_features,
            "pos": featurizer.convert_to_pos_features,
            "mlm": featurizer.convert_to_mlm_features,
            "dep": featurizer.convert_to_dependency_parsing_features,
        }

        self.columns_dict = {
            "ner": ["input_ids", "attention_mask", "labels", "task_type"],
            "pos": ["input_ids", "attention_mask", "labels", "task_type"],
            "dep": ["input_ids", "attention_mask", "word_starts", "labels_arcs", "labels_rels", "task_type"],
            "mlm": ["input_ids", "attention_mask", 'special_tokens_mask', "task_type"],
        }

        self.label_column_name_dict = {
            "ner": ['labels'],
            "pos": ['labels'],
            'mlm': ['##_NO_LABEL_##'],
            'dep': ["labels_arcs", "labels_rels"],
        }

        self.wikiann_langs = ['ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast', 'ay', 'az', 'ba',
                              'bar', 'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 'bo', 'br', 'bs', 'ca', 'cbk-zam',
                              'cdo', 'ce', 'ceb', 'ckb', 'co', 'crh', 'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv',
                              'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr',
                              'fur', 'fy', 'ga', 'gan', 'gd', 'gl', 'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 'hu',
                              'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn',
                              'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij', 'lmo', 'ln', 'lt', 'lv', 'map-bms', 'mg',
                              'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds',
                              'ne', 'nl', 'nn', 'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms', 'pnb', 'ps',
                              'pt', 'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 'simple',
                              'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tk', 'tl',
                              'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu',
                              'xmf', 'yi', 'yo', 'zea', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue']

        self.label_list_dict = {
            'ner': ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
            'pos': ["NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", "PROPN", "PRON", "X",
                    "_", "ADV", "INTJ", "VERB", "AUX"],
            'dep': UD_HEAD_LABELS
        }

        self.lang_to_ud = {
            'en': 'en_ewt',
            'ar': 'ar_padt',
            'zh': 'zh_gsd',
            'tr': 'tr_imst',
            'kk': 'kk_ktb',
            'ta': 'ta_ttb',
            'zh-yue': 'yue_hk',
            'br': 'br_keb',
            'is': 'is_pud',
            'tl': 'tl_trg',
            'yo': 'yo_ytb',
            'fo': 'fo_oft',
            'ug': 'ug_udt',
            'mt': 'mt_mudt',
            'sa': 'sa_ufal',
            'hsb': 'hsb_ufal',
            'gn': 'gun_thomas',
        }

        self.hpx_eval = Eval(self.label_list_dict)

    def setup_datasets(self):

        dataset_dict = dict()
        for task_name in self.tasks:
            if task_name == 'ner':
                dataset_dict[task_name] = {lang_name: load_dataset('wikiann', lang_name)
                                           for lang_name in self.languages if lang_name in self.wikiann_langs}
            elif task_name == 'pos':
                dataset_dict[task_name] = {lang_name: load_dataset('universal_dependencies', self.lang_to_ud[lang_name])
                                           for lang_name in self.languages if lang_name in self.lang_to_ud}
            elif task_name == 'dep':
                dataset_dict[task_name] = {lang_name: load_dataset('universal_dependencies', self.lang_to_ud[lang_name])
                                           for lang_name in self.languages if lang_name in self.lang_to_ud}
            elif task_name == 'mlm':
                dataset_dict[task_name] = {lang_name: load_dataset('text',
                                                                   data_files=f'{self.args.data_folder}/mlm/{lang_name}.txt')
                                           for lang_name in self.languages
                                           if os.path.isfile(f'{self.args.data_folder}/mlm/{lang_name}.txt')}
        return dataset_dict

    def setup_model(self):
        hpx_model = Hyperx.create(
            model_name=self.model_name,
            model_type_dict={
                task_name: self.task_model_dict[task_name] for task_name in self.tasks
            },
            model_config_dict={
                task_name: self.task_conf_dict[task_name] for task_name in self.tasks
            },
            hpx_args=self.args
        )
        return hpx_model
