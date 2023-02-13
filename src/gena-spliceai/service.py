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
class SpliceAIConf:
    working_segment = 5000
    segment_step = None
    batch_size = 4
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'gena_lm.modeling_bert:BertForTokenClassification'
    model_cfg = service_folder.joinpath('data/configs/L12-H768-A12-V32k-preln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')


class SpliceAIPreprocess:
    def __init__(self, tokenizer, max_seq_len=512, targets_offset=5000, targets_len=5000):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.targets_offset = targets_offset
        self.targets_len = targets_len
        # load default target value
        def_val = service_folder.joinpath('data/checkpoints/def_target_val.npy')
        self.default_target_value = np.load(str(def_val))

    def tokenize_inputs(self, seq):
        depad_seq_l = seq.strip("N")

        if len(depad_seq_l) > self.targets_offset:
            depad_seq_l = seq[:self.targets_offset]

        if len(depad_seq_l) < self.targets_len:
            print(f"ATTENTION! The length of the incoming sequence is less than 5,000 nucleotides; the model's "
                  f"response under given conditions may not be entirely correct.")

        mid_encoding = self.tokenizer(depad_seq_l,
                                      add_special_tokens=False,
                                      padding=False,
                                      return_offsets_mapping=True,
                                      return_tensors="np")

        assert np.all(mid_encoding["attention_mask"][0] == 1)
        assert np.all(mid_encoding["token_type_ids"][0] == 0)

        token_type_ids = np.zeros(shape=self.max_seq_len, dtype=np.int64)
        n_service_tokens = 4  # CLS-left-SEP-mid-SEP-right-SEP (PAD)
        L_mid = len(mid_encoding["input_ids"][0])

        # case I. target's encoding >= max_seq_len; don't add context & trim target if needed
        if L_mid + n_service_tokens > self.max_seq_len:
            # st = (L_mid // 2) - (self.max_seq_len - n_service_tokens) // 2
            # en = st + (self.max_seq_len - n_service_tokens)
            st = 0
            en = self.max_seq_len - n_service_tokens

            input_ids = np.concatenate(
                [
                    [
                        self.tokenizer.convert_tokens_to_ids("[CLS]"),
                        self.tokenizer.convert_tokens_to_ids("[SEP]"),
                    ],
                    mid_encoding["input_ids"][0][st:en],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")] * 2,
                ]
            )

        # case II. target+context encoding < max_seq_len, we need to pad
        elif L_mid + n_service_tokens < self.max_seq_len:
            n_pads = self.max_seq_len - (L_mid + n_service_tokens)
            input_ids = np.concatenate(
                [
                    [
                        self.tokenizer.convert_tokens_to_ids("[CLS]"),
                        self.tokenizer.convert_tokens_to_ids("[SEP]"),
                    ],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[PAD]")] * n_pads,
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")]
                ]
            )

        # case III. target == max_seq_len
        elif L_mid + n_service_tokens == self.max_seq_len:
            input_ids = np.concatenate(
                [
                    [self.tokenizer.convert_tokens_to_ids("[CLS]")],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )

        else:
            raise ValueError("Unexpected encoding length")

        assert len(input_ids) == self.max_seq_len

        attention_mask = np.array(input_ids != self.tokenizer.pad_token_id, dtype=np.int64)

        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask}

    def __call__(self, seq: str):
        return self.tokenize_inputs(seq)


class SpliceaiService:
    def __init__(self, config: dataclass):
        self.conf = config
        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.prepocesser = SpliceAIPreprocess(self.tokenizer)

        # define model | labels: 0, 1, 2; multi-class multi-label classification
        model_cfg = AutoConfig.from_pretrained(config.model_cfg)
        model_cfg.num_labels = 3
        model_cfg.problem_type = 'multi_label_classification'
        model_cls = get_cls_by_name(config.model_cls)
        self.model = model_cls(config=model_cfg)

        # load weights
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        missing_k, unexpected_k = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if len(missing_k) != 0:
            print(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            print(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

        # run eval mode
        self.model.eval()
        self.model_forward_args = set(inspect.getfullargspec(self.model.forward).args)

        # label counts in test set: [8378616.,    9842.,   10258.]) upweight class 1 and 2
        self.pos_weight = torch.tensor([1.0, 100.0, 100.0])

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
            batch.append(self.prepocesser(dna_seq))

        # model inference
        batch = self.create_batch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        # postprocessing
        service_response = dict()
        # write predictions
        predictions = torch.sigmoid(model_out['logits']).detach().numpy()  # [bs, seq, 3]
        service_response['acceptors'] = np.where(predictions[..., 1] > 0.5, 1, 0).flatten()  # [bs*seq]
        service_response['donors'] = np.where(predictions[..., 2] > 0.5, 1, 0).flatten()  # [bs*seq]

        # write tokens
        input_ids = batch['input_ids'].detach().numpy().flatten()  # [bs*seq]
        service_response['seq'] = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        # for batch_element in input_ids:
        #     service_response['seq'].append()

        return service_response
