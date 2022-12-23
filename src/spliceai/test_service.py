import inspect
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from src import service_folder
from src.gena_lm.utils import get_cls_by_name


@dataclass
class SpliceAIConf:
    input_len = 5000
    context_len = 5000
    max_seq_len = 512
    tokenizer = service_folder.joinpath('data/tokenizers/human/BPE_32k/')
    model_cls = 'src.gena_lm.modeling_bert:BertForTokenClassification'
    model_cfg = service_folder.joinpath('data/configs/L12-H768-A12-V32k-preln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/spliceai/model_best.pth')


class SpliceaiService:
    def __init__(self, config: dataclass):
        self.conf = config
        self.input_seq_len = config.input_len + (2 * config.context_len)

        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        # define model | labels: 0, 1, 2; multi-class multi-label classification
        model_cfg = AutoConfig.from_pretrained(config.model_cfg)
        model_cfg.num_labels = 3
        model_cfg.problem_type = 'multi_label_classification'
        model_cls = get_cls_by_name(config.model_cls)
        self.model = model_cls(config=model_cfg)
        self.model.load_state_dict(
            torch.load(config.checkpoint_path, map_location=torch.device('cpu')),
            strict=False)
        self.model.eval()
        self.model_forward_args = set(inspect.getfullargspec(self.model.forward).args)

    @staticmethod
    def dna_string_padding(seq: str, length: int):
        pad_len = (length - len(seq)) // 2
        seq = ('N' * pad_len) + seq + ('N' * pad_len)
        if len(seq) < length:
            seq += 'N' * (length - len(seq))

        return seq

    def preprocess(self, dna_seq: str):
        batch = []
        if len(dna_seq) < self.conf.input_len:
            dna_seq = ('N' * self.conf.context_len) + dna_seq + ('N' * self.conf.context_len)
            batch.append(dna_seq)

        elif (len(dna_seq) > self.conf.input_len) and (len(dna_seq) < self.input_seq_len):
            batch.append(self.dna_string_padding(dna_seq, self.input_seq_len))

        elif len(dna_seq) > self.input_seq_len:
            offset = len(dna_seq) // self.conf.context_len
            if offset == 3:
                batch.append(dna_seq[:self.input_seq_len])
            elif offset > 3:
                for i in range(3, offset, 1):
                    batch.append(dna_seq[(i * self.conf.context_len):((i + 1) * self.conf.context_len)])

        return batch

    def tokenize_inputs(self, seq: str):
        left, mid, right = (seq[:self.conf.context_len],
                            seq[self.conf.context_len:self.conf.context_len + self.conf.input_len],
                            seq[self.conf.context_len + self.conf.input_len:])

        mid_encoding = self.tokenizer(mid,
                                      add_special_tokens=False,
                                      padding=False,
                                      return_offsets_mapping=True,
                                      return_tensors="np")

        context_encoding = self.tokenizer(left + "X" + right,
                                          add_special_tokens=False,
                                          padding=False,
                                          return_offsets_mapping=True,
                                          return_tensors="np")

        for encoding in [mid_encoding, context_encoding]:
            assert np.all(encoding["attention_mask"][0] == 1)
            assert np.all(encoding["token_type_ids"][0] == 0)

        token_type_ids = np.zeros(shape=self.conf.max_seq_len, dtype=np.int64)

        boundary_pos = int(np.where(context_encoding["offset_mapping"][0] == [len(left), len(left) + 1])[0][0])

        boundary_token = context_encoding["input_ids"][0][boundary_pos].tolist()
        assert (self.tokenizer.convert_ids_to_tokens(boundary_token) == "[UNK]"), \
            "Error during context tokens processing"

        n_service_tokens = 4  # CLS-left-SEP-mid-SEP-right-SEP (PAD)

        l_mid = len(mid_encoding["input_ids"][0])
        l_left = boundary_pos
        l_right = len(context_encoding["token_type_ids"][0]) - l_left - 1

        # case I. target's encoding >= max_seq_len; don't add context & trim target if needed
        if l_mid + n_service_tokens >= self.conf.max_seq_len:
            # st = (l_mid // 2) - (self.max_seq_len - n_service_tokens) // 2
            # en = st + (self.max_seq_len - n_service_tokens)
            st = 0
            en = self.conf.max_seq_len - n_service_tokens

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
        elif l_mid + l_left + l_right + n_service_tokens <= self.conf.max_seq_len:
            n_pads = self.conf.max_seq_len - (l_mid + l_left + l_right + n_service_tokens)
            input_ids = np.concatenate(
                [
                    [self.tokenizer.convert_tokens_to_ids("[CLS]")],
                    context_encoding["input_ids"][0][:boundary_pos],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    context_encoding["input_ids"][0][boundary_pos + 1:],
                    [self.tokenizer.convert_tokens_to_ids("[PAD]")] * n_pads,
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )

        # case III. target+context encoding > max_seq_len, we need to trim
        elif l_mid + l_left + l_right + n_service_tokens > self.conf.max_seq_len:
            # compute trimming. The aims are to
            # a) make the total length == self.max_seq_len
            # b) make the left and right context size as close to each other as possible
            oversize = l_mid + l_left + l_right + n_service_tokens - self.conf.max_seq_len
            if l_left >= l_right:
                trim_left = oversize / 2.0 + min(
                    (l_left - l_right) / 2.0, oversize / 2.0
                )
                trim_right = max(0, (oversize - (l_left - l_right)) / 2.0)
            else:
                trim_right = oversize / 2.0 + min(
                    (l_right - l_left) / 2.0, oversize / 2.0
                )
                trim_left = max(0, (oversize - (l_right - l_left)) / 2.0)
            assert (int(trim_right) == trim_right) == (int(trim_left) == trim_left)
            if int(trim_right) != trim_right:
                trim_left += 0.5
                trim_right -= 0.5
            assert (int(trim_right) - trim_right) == (int(trim_left) - trim_left) == 0
            assert oversize == trim_left + trim_right

            trim_left = int(trim_left)
            trim_right = int(trim_right)

            input_ids = np.concatenate(
                [
                    [self.tokenizer.convert_tokens_to_ids("[CLS]")],
                    context_encoding["input_ids"][0][trim_left:boundary_pos],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    mid_encoding["input_ids"][0],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                    context_encoding["input_ids"][0][boundary_pos + 1: l_left + l_right + 1 - trim_right],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )

        else:
            raise ValueError("Unexpected encoding length")

        attention_mask = np.array(input_ids != self.tokenizer.pad_token_id, dtype=np.int64)

        return input_ids, token_type_ids, attention_mask

    def create_batch(self, examples: List[str]):
        batch = {'input_ids': [],
                 'token_type_ids': [],
                 'attention_mask': [],
                 'labels': None,
                 "labels_ohe": None,
                 'labels_mask': None}

        for dna_seq in examples:
            input_ids, token_type_ids, attention_mask = self.tokenize_inputs(dna_seq)
            batch['input_ids'].append(input_ids)
            batch['token_type_ids'].append(token_type_ids)
            batch['attention_mask'].append(attention_mask)

        batch['input_ids'] = torch.from_numpy(np.vstack(batch['input_ids'])).int()
        batch['token_type_ids'] = torch.from_numpy(np.vstack(batch['token_type_ids'])).int()
        batch['attention_mask'] = torch.from_numpy(np.vstack(batch['attention_mask'])).float()

        return batch

    def __call__(self, dna_examples) -> Dict:
        batch = self.preprocess(dna_examples)
        batch = self.create_batch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        predictions = np.argmax(torch.sigmoid(model_out['logits']).detach().numpy(), axis=2).flatten()
        input_ids = batch['input_ids'].detach().numpy().flatten()
        assert len(predictions) == len(input_ids)

        service_responce = dict()
        service_responce['prediction'] = predictions
        service_responce['seq'] = self.tokenizer.convert_ids_to_tokens(input_ids)

        return service_responce


if __name__ == "__main__":
    conf = SpliceAIConf()
    instance_class = SpliceaiService(conf)

    example = 'CTCGTTCCGCGCCCGCCATGGAACCGGATGTACGTTATAGCTATTACGCTACTGTGGGTGCACTCGTTCCGCGCCCGCCATGGAACCGGATGGTCTAGCCGATCTGACGCTCGTTCCGCGCCCGCCATGGAACCGGATGCCCCGCCCCTGGTTTCGAGTCGCTGGCCTGCTGGGTGTCATCGCATTATCGATATTGCATTACGTTATAGCTATTACCTCGTTCCGCGCCCGCCATGGAACCGGATGGCTACTGTGGGTGCAGTCTAGC'
    y = instance_class(example)

    print(y['prediction'].shape)
    print(y['prediction'])
    print(len(y['seq']), y['seq'])
