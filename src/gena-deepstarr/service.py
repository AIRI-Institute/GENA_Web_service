import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, ClassVar

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
import pandas as pd 
from captum.attr import LayerIntegratedGradients # type: ignore


from gena_lm.utils import get_cls_by_name
import logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

service_folder = Path(__file__).parent.absolute()



@dataclass
class DeepStarrConf:
    working_segment = 249
    segment_step = 249
    batch_size = 1
    max_seq_len = 128
    # model parameters
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cls = 'gena_lm.modeling_bert:BertForSequenceClassification'
    model_cfg = service_folder.joinpath('data/configs/L12-H768-A12-V32k-preln.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')
    base_model = "bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16"
    max_tokens: int = 512
    attr_steps: int = 10


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

        for fn in features:
            features[fn] = features[fn][0]


        return features


def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

def token_positions(seq_tok):
    if len(seq_tok._encodings) != 1:
        raise Exception("Unexpected number of encodings")
    
    prev_e = 0
    ind2pos = []
    for ind, (_, e) in enumerate(seq_tok._encodings[0].offsets):
        ind2pos.append( (prev_e, e))
        prev_e = e
    return ind2pos

class DeepStarrService:
    DEV_PRED_INDEX: int = 0
    HK_PRED_INDEX: int = 1

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

    def __call__(self, queries, temp_storage: Path, calc_importance: bool) -> Dict:
        # preprocessing
        samples = []
        for q in queries:
            samples.append(self.preprocessor(q['seq']))

        # model inference
        batch = self.create_batch(samples)
        with torch.inference_mode():
            model_out = self.model(**{k: batch[k] for k in batch if k in self.model_forward_args})

        # postprocessing
        service_response = dict()
        # write predictions
        predictions = model_out['logits'].detach().numpy()  # [bs, 2]
        service_response['dev'] = predictions[:, self.DEV_PRED_INDEX]
        service_response['hk'] = predictions[:, self.HK_PRED_INDEX]
        # write tokens
        service_response['seq'] = []
        input_ids = batch['input_ids'].detach().numpy()
        for batch_element in input_ids:
            service_response['seq'].append(self.tokenizer.convert_ids_to_tokens(batch_element,
                                                                                skip_special_tokens=True))
            
        if calc_importance:
            attributions = self.annotate_predictions(samples, predictions, queries, temp_storage)
            service_response['attr'] = attributions
            
        service_response['queries'] = queries
        return service_response
    
    def create_attr_object(self):
        def predict_regressor(inputs, 
                               token_type_ids=None, 
                               attention_mask=None):
            assert token_type_ids is not None
            assert attention_mask is not None
            output = self.model(inputs, 
                   token_type_ids=token_type_ids, 
                   attention_mask=attention_mask)
            return output.logits
        lig_object = LayerIntegratedGradients(predict_regressor, self.model.bert.embeddings)
        return lig_object

    def annotate_predictions(self, samples, preds, dna_queries, temp_storage):
        lig_object = self.create_attr_object()
        attributions = []
        for si, smpl in enumerate(samples):
            smpl_attrs = {}
            pred = preds[si]
            for pred_name, pred_ind in ( ('dev', self.DEV_PRED_INDEX),
                                         ('hk', self.HK_PRED_INDEX)): 
                logger.info(f"Creating attributions for sample {si}")
                attr = self.annotate_sample(lig_object=lig_object, 
                                            sample=smpl, 
                                            target=pred_ind, 
                                            query=dna_queries[si])
                temp_path = temp_storage / f"attr_sample{si}_target{pred_ind}"
                attr.to_csv(temp_path, sep="\t", index=False)
                smpl_attrs[pred_name] = temp_path

            attributions.append(smpl_attrs)
        return attributions

    def annotate_sample(self, lig_object, sample, target, query):
        presample = sample
        sample = {
           "input_ids": torch.LongTensor(sample['input_ids']).unsqueeze(0),
           "token_type_ids": torch.LongTensor(sample['token_type_ids']).unsqueeze(0),
           "attention_mask": torch.LongTensor(sample['attention_mask']).unsqueeze(0),
        }

        attributions = lig_object.attribute(inputs=(sample['input_ids'],
                                            sample['token_type_ids'],
                                            sample['attention_mask']),
                                    target=int(target),
                                    n_steps=self.conf.attr_steps,
                                    return_convergence_delta=False)
        

        attributions = attributions[:, 1:-1, :] # remove CLS and SEP
        attributions = summarize_attributions(attributions).cpu()

        bed_like_table = {'tok_pos': [], 'token': [], 'attr': [], 'start': [], 'end': []}
        
        pretokens = self.tokenizer.convert_ids_to_tokens(presample['input_ids'], skip_special_tokens=False)
        tokens = pretokens

        startends = token_positions(presample)
        
        for i, tok in enumerate(tokens):
            start, end = startends[i]
            if start >= end:
                continue # special token 
            attr = attributions[i].item()
            bed_like_table['tok_pos'].append(i)
            bed_like_table['token'].append(tok)
            bed_like_table['attr'].append(attr)
           
            bed_like_table['start'].append(start)
            bed_like_table['end'].append(end)

        seq_len = len(query['seq'])
        df = pd.DataFrame(bed_like_table)
        
        df = df[np.logical_and(df['start'] >= query['lpad'],
                               df['end'] <= (seq_len - query['rpad'])
                              )]
        df['start'] = df['start'] + query['context_start']
        df['end'] = df['end'] + query['context_start']
        
        return df 
