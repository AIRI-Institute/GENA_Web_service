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
class DeepStarrConf:
    working_segment = 249
    segment_step = None
    batch_size = 16
    max_seq_len = 128
    # model parameters
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'gena_lm.modeling_bert:BertForSequenceClassification'
    model_cfg = service_folder.joinpath('data/configs/L12-H768-A12-V32k-preln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')
    base_model = "bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16"


class DeepStarrPreprocessor:
    def __init__(self, tokenizer, max_seq_len: int, pad_to_max: bool = True, truncate: str = 'right'):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max
        self.truncate = truncate

    def __call__(self, dna_seq: str):
        features = self.tokenizer(dna_seq,
                                  add_special_tokens=True,
                                  padding="max_length",
                                  truncation="longest_first",
                                  max_length=self.max_seq_len,
                                  return_tensors="np")

        features = {"input_ids": features["input_ids"][0],
                    "token_type_ids": features["token_type_ids"][0],
                    "attention_mask": features["attention_mask"][0]}

        return {**features}


class DeepStarrService:
    def __init__(self, config: dataclass):
        self.conf = config

        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        # define preprocessor
        self.preprocessor = DeepStarrPreprocessor(tokenizer=self.tokenizer, max_seq_len=config.max_seq_len)

        # define model | sequence binary classification
        model_cfg = AutoConfig.from_pretrained(config.model_cfg)

        # regression task with two targets
        model_cfg.num_labels = 2
        model_cfg.problem_type = 'regression'
        model_cls = get_cls_by_name(config.model_cls)
        print(f'[ Using model class: {model_cls} ]')
        self.model = model_cls(config=model_cfg)

        # load model checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model_forward_args = set(inspect.getfullargspec(self.model.forward).args)

    @staticmethod
    def create_batch(seq_list: List[Dict]) -> Dict:
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

    def __call__(self, dna_examples: List[str]) -> Dict:
        # preprocessing
        batch = []
        for dna_seq in dna_examples:
            batch.append(self.preprocessor(dna_seq))

        # model inference
        batch = self.create_batch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        # postprocessing
        service_response = dict()
        # write predictions
        predictions = model_out['logits'].detach().numpy()  # [bs, 2]
        service_response['dev'] = predictions[:, 0]
        service_response['hk'] = predictions[:, 1]
        # write tokens
        service_response['seq'] = []
        input_ids = batch['input_ids'].detach().numpy()
        for batch_element in input_ids:
            service_response['seq'].append(self.tokenizer.convert_ids_to_tokens(batch_element,
                                                                                skip_special_tokens=True))

        return service_response
