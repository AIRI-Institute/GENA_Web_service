import inspect
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from src.gena_lm.utils import get_cls_by_name
from transformers import AutoConfig, AutoTokenizer

from src import service_folder


@dataclass
class SpliceAIConf:
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'src.gena_lm.modeling_bert:BertForTokenClassification'
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
        self.default_target_value = np.load(def_val)

    @staticmethod
    def get_token_classes(seq_encoding, target, targets_offset):
        tokens_info = pd.DataFrame.from_records(
            seq_encoding["offset_mapping"][0], columns=["st", "en"]
        )
        tokens_info["length"] = tokens_info["en"] - tokens_info["st"]

        # handle special tokens which can be anywhere in the seq with the offeset (0,0)
        nonzero_length_mask = tokens_info["length"].values > 0

        # all other tokens should have ascending token_start coordinate
        assert np.all(
            tokens_info[nonzero_length_mask]["st"].values[:-1]
            <= tokens_info[nonzero_length_mask]["st"].values[1:]
        )

        # fill target class information
        target_field_names = []
        for target_class in [1, 2]:
            target_field_name = "class_" + str(target_class)
            target_field_names.append(target_field_name)
            tokens_info[target_field_name] = 0

            nonzero_target_positions = (
                    np.where(target == target_class)[0] + targets_offset
            )  # non-zero target coordinates
            nonzero_target_token_ids = (
                    np.searchsorted(
                        tokens_info[nonzero_length_mask]["st"],
                        nonzero_target_positions,
                        side="right",
                    )
                    - 1
            )  # ids of tokens
            # containing non-zero targets
            # in sequence coordinate system
            nonzero_target_token_ids = (
                tokens_info[nonzero_length_mask].iloc[nonzero_target_token_ids].index
            )
            tokens_info.loc[nonzero_target_token_ids, target_field_name] = target_class
            # tokens_info.loc[nonzero_target_token_ids, target_field_name] = 1
            # fill all service tokens with -100
            tokens_info.loc[~nonzero_length_mask, target_field_name] = -100

            # fill context tokens with -100
            target_first_token = (
                    np.searchsorted(
                        tokens_info[nonzero_length_mask]["st"],
                        targets_offset,
                        side="right",
                    )
                    - 1
            )

            mask_ids = (
                tokens_info.loc[nonzero_length_mask, :].iloc[:target_first_token].index
            )
            tokens_info.loc[mask_ids, target_field_name] = -100

            target_last_token = (
                    np.searchsorted(
                        tokens_info[nonzero_length_mask]["st"],
                        targets_offset + len(target) - 1,
                        side="right",
                    )
                    - 1
            )
            if target_last_token + 1 < len(
                    tokens_info.loc[nonzero_length_mask, target_field_name]
            ):
                target_last_token += 1
                mask_ids = tokens_info.loc[nonzero_length_mask, :][target_last_token:].index
                tokens_info.loc[mask_ids, target_field_name] = -100
        return tokens_info[target_field_names].values

    def tokenize_inputs(self, seq, target):
        depad_seq_l = seq.lstrip("N")
        if self.targets_offset - (len(seq) - len(depad_seq_l)) < 0:
            depad_seq_l = seq[self.targets_offset:]
        targets_offset = self.targets_offset - (len(seq) - len(depad_seq_l))

        assert targets_offset >= 0
        if targets_offset + self.targets_len <= len(depad_seq_l):
            print(f"ATTENTION! The length of the incoming sequence is less than 10,000 nucleotides; the model's "
                  f"response under given conditions may not be entirely correct.")

        depad_seq_both = depad_seq_l.strip("N")
        if targets_offset + self.targets_len > len(depad_seq_both):
            seq = depad_seq_l[:targets_offset + self.targets_len]
        else:
            seq = depad_seq_both

        left, mid, right = (
            seq[:targets_offset],
            seq[targets_offset: targets_offset + self.targets_len],
            seq[targets_offset + self.targets_len:],
        )

        mid_encoding = self.tokenizer(
            mid,
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        context_encoding = self.tokenizer(
            left + "X" + right,
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )

        for encoding in [mid_encoding, context_encoding]:
            assert np.all(encoding["attention_mask"][0] == 1)
            assert np.all(encoding["token_type_ids"][0] == 0)

        labels = self.get_token_classes(mid_encoding, target, 0)
        token_type_ids = np.zeros(shape=self.max_seq_len, dtype=np.int64)

        boundary_pos = int(
            np.where(
                context_encoding["offset_mapping"][0] == [len(left), len(left) + 1]
            )[0][0]
        )
        boundary_token = context_encoding["input_ids"][0][boundary_pos].tolist()
        assert (
                self.tokenizer.convert_ids_to_tokens(boundary_token) == "[UNK]"
        ), "Error during context tokens processing"

        n_service_tokens = 4  # CLS-left-SEP-mid-SEP-right-SEP (PAD)

        L_mid = len(mid_encoding["input_ids"][0])
        L_left = boundary_pos
        L_right = len(context_encoding["token_type_ids"][0]) - L_left - 1

        # case I. target's encoding >= max_seq_len; don't add context & trim target if needed
        if L_mid + n_service_tokens >= self.max_seq_len:
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
            labels = np.concatenate(
                [[[-100, -100]] * 2, labels[st:en], [[-100, -100]] * 2]
            )
        # case II. target+context encoding < max_seq_len, we need to pad
        elif L_mid + L_left + L_right + n_service_tokens <= self.max_seq_len:
            n_pads = self.max_seq_len - (L_mid + L_left + L_right + n_service_tokens)
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
            labels = np.concatenate(
                [
                    [[-100, -100]],
                    [[-100, -100]] * (boundary_pos + 1),
                    labels,
                    [[-100, -100]]
                    * (len(context_encoding["input_ids"][0]) - boundary_pos - 1 + 1),
                    [[-100, -100]] * (n_pads + 1)
                ]
            )
        # case III. target+context encoding > max_seq_len, we need to trim
        elif L_mid + L_left + L_right + n_service_tokens > self.max_seq_len:
            # compute trimming. The aims are to
            # a) make the total length == self.max_seq_len
            # b) make the left and right context size as close to each other as possible
            oversize = L_mid + L_left + L_right + n_service_tokens - self.max_seq_len
            if L_left >= L_right:
                trim_left = oversize / 2.0 + min(
                    (L_left - L_right) / 2.0, oversize / 2.0
                )
                trim_right = max(0, (oversize - (L_left - L_right)) / 2.0)
            else:
                trim_right = oversize / 2.0 + min(
                    (L_right - L_left) / 2.0, oversize / 2.0
                )
                trim_left = max(0, (oversize - (L_right - L_left)) / 2.0)
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
                    context_encoding["input_ids"][0][
                    boundary_pos + 1: L_left + L_right + 1 - trim_right
                    ],
                    [self.tokenizer.convert_tokens_to_ids("[SEP]")],
                ]
            )
            labels = np.concatenate(
                [
                    [[-100, -100]],
                    [[-100, -100]] * (boundary_pos - trim_left + 1),
                    labels,
                    [[-100, -100]],
                    [[-100, -100]] * (L_right - trim_right + 1),
                ]
            )
        else:
            raise ValueError("Unexpected encoding length")

        assert len(input_ids) == len(labels) == self.max_seq_len

        # convert labels to (seq_len, n_labels) shape
        n_labels = 3
        labels_ohe = np.zeros((len(labels), n_labels))
        for label in range(n_labels):
            labels_ohe[(labels == label).max(axis=-1), label] = 1.0

        # set mask to 0.0 for tokens with no labels, these examples should not be used for loss computation
        labels_mask = np.ones(len(labels))
        labels_mask[labels_ohe.sum(axis=-1) == 0.0] = 0.0

        attention_mask = np.array(input_ids != self.tokenizer.pad_token_id,
                                  dtype=np.int64
                                  )
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_ohe": labels_ohe,
            "labels_mask": labels_mask,
        }

    def __call__(self, seq: str):
        return self.tokenize_inputs(seq, self.default_target_value)


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

    def create_batch(self, batch: Dict):
        # todo: переписать для работы с батчами
        seq_len = batch['input_ids'].shape[0]
        bs = 1
        res_dict = {'input_ids': torch.from_numpy(batch['input_ids'])[None, ...].int(),
                    'token_type_ids': torch.from_numpy(batch['token_type_ids'])[None, ...].int(),
                    'attention_mask': torch.from_numpy(batch['attention_mask'])[None, ...].float(),
                    'labels': torch.from_numpy(batch['labels_ohe'])[None, ...].float(),
                    'labels_mask': torch.from_numpy(batch['labels_mask'])[None, ...].float(),
                    'pos_weight': self.pos_weight.repeat(bs, seq_len, 1)}

        return res_dict

    def __call__(self, dna_example: str) -> Dict:
        batch = self.prepocesser(dna_example)
        batch = self.create_batch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        input_ids = batch['input_ids'].detach().numpy().flatten()
        predictions = torch.sigmoid(model_out['logits']).detach().numpy()

        service_responce = dict()
        service_responce['acceptors'] = np.where(predictions[..., 1] > 0.5, 1, 0)
        service_responce['donors'] = np.where(predictions[..., 2] > 0.5, 1, 0)
        service_responce['seq'] = self.tokenizer.convert_ids_to_tokens(input_ids)

        return service_responce
