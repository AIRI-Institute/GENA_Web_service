import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from gena_lm.utils import get_cls_by_name

service_folder = Path(__file__).parent.absolute()


@dataclass
class PromotersConf:
    bpe_dropout = 0.0
    working_segment = 2000
    segment_step = 1000
    batch_size = 32
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'src.gena_lm.modeling_bert:BertForSequenceClassification'
    model_cfg = service_folder.joinpath('data/configs/L24-H1024-A16-V32k-preln-lastln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')
    base_model = "bert_large_512_lastln_t2t_1000G_bs256_lr_1e-04_fp16-1750k_iters"


class PromotersPreprocessor:
    def __init__(self, tokenizer, max_seq_len=512, pad_to_max=True, truncate='right'):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max
        self.truncate = truncate

    def get_features(self, x):
        tokens = self.tokenizer.tokenize(x)

        if self.truncate == 'right':
            tokens = tokens[:self.max_seq_len - 2]
        elif self.truncate == 'left':
            tokens = tokens[-(self.max_seq_len - 2):]
        elif self.truncate == 'mid':
            mid = len(tokens) // 2
            left_ctx = (self.max_seq_len - 2) // 2
            right_ctx = (self.max_seq_len - 2) - left_ctx
            tokens = tokens[max(0, mid - left_ctx): min(mid + right_ctx, len(tokens))]

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        token_type_ids = [0] * seq_len
        attention_mask = [1] * seq_len

        if self.pad_to_max:
            input_ids += [self.tokenizer.pad_token_id] * max(self.max_seq_len - seq_len, 0)
            token_type_ids += [0] * max(self.max_seq_len - seq_len, 0)
            attention_mask += [0] * max(self.max_seq_len - seq_len, 0)

        return {'input_ids': np.array(input_ids),
                'token_type_ids': np.array(token_type_ids),
                'attention_mask': np.array(attention_mask)}

    def __call__(self, dna_seq: str):
        features = self.get_features(dna_seq)
        return {**features}


class PromotersService:
    def __init__(self, config: dataclass):
        self.conf = config
        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if (config.bpe_dropout is not None) and (config.bpe_dropout > 0.0):
            if hasattr(self.tokenizer._tokenizer.model, 'dropout'):
                self.tokenizer._tokenizer.model.dropout = config.bpe_dropout
            else:
                print('BPE dropout is not set as tokenizer does not support it.')

        # define preprocessor
        self.preprocessor = PromotersPreprocessor(tokenizer=self.tokenizer)

        # define model | sequence binary classification
        model_cfg = AutoConfig.from_pretrained(config.model_cfg)
        model_cfg.num_labels = 2
        model_cls = get_cls_by_name(config.model_cls)
        print(f'[ Using model class: {model_cls} ]')
        self.model = model_cls(config=model_cfg)

        # load model checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model_forward_args = set(inspect.getfullargspec(self.model.forward).args)

    @staticmethod
    def create_batch(seq_list: List[Dict]):
        batch = {'input_ids': [],
                 'token_type_ids': [],
                 'attention_mask': [],
                 'labels': None,
                 "labels_ohe": None,
                 'labels_mask': None}

        for features in seq_list:
            batch['input_ids'].append(features['input_ids'])
            batch['token_type_ids'].append(features['token_type_ids'])
            batch['attention_mask'].append(features['attention_mask'])

        batch['input_ids'] = torch.from_numpy(np.vstack(batch['input_ids'])).int()
        batch['token_type_ids'] = torch.from_numpy(np.vstack(batch['token_type_ids'])).int()
        batch['attention_mask'] = torch.from_numpy(np.vstack(batch['attention_mask'])).float()

        return batch

    def __call__(self, dna_examples: List[str]):
        # preprocessing
        batch = []
        for dna_seq in dna_examples:
            batch.append(self.preprocessor(dna_seq))

        # model inference
        batch = self.create_batch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        # postprocessing
        service_response = dict()
        input_ids = batch['input_ids'].detach().numpy().flatten()
        predictions = torch.argmax(model_out['logits'].detach(), dim=-1).numpy()
        service_response['prediction'] = predictions

        service_response['seq'] = []
        for batch_element in input_ids:
            service_response['seq'].append(self.tokenizer.convert_ids_to_tokens(batch_element,
                                                                                skip_special_tokens=True))

        return service_response
