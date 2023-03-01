import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoConfig

from gena_lm.utils import get_cls_by_name

service_folder = Path(__file__).parent.absolute()


@dataclass
class SpliceAIConf:
    max_seq_len = 512
    segment_step = None
    batch_size = 4
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'gena_lm.modeling_bert:BertForTokenClassification'
    model_cfg = service_folder.joinpath('data/configs/L12-H768-A12-V32k-preln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')


class SpliceaiService:
    def __init__(self, config: dataclass):
        self.conf = config
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
    def batch2torch(batch: Dict) -> Dict:
        batch['labels'] = None
        batch['labels_ohe'] = None
        batch['labels_mask'] = None

        batch['input_ids'] = torch.from_numpy(np.vstack(batch['input_ids'])).int()
        batch['token_type_ids'] = torch.from_numpy(np.vstack(batch['token_type_ids'])).int()
        batch['attention_mask'] = torch.from_numpy(np.vstack(batch['attention_mask'])).float()

        return batch

    def __call__(self, batch: Dict) -> Dict:
        # model inference
        batch = self.batch2torch(batch)
        model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        # postprocessing
        service_response = dict()
        # write tokens
        service_response['input_ids'] = batch['input_ids'].detach().numpy().flatten()  # [bs*seq]
        # write predictions
        predictions = torch.sigmoid(model_out['logits']).detach().numpy()  # [bs, seq, 3]
        service_response['acceptors'] = np.where(predictions[..., 1] > 0.5, 1, 0).flatten()  # [bs*seq]
        service_response['donors'] = np.where(predictions[..., 2] > 0.5, 1, 0).flatten()  # [bs*seq]

        return service_response
