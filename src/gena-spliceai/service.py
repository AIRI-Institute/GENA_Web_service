import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import BigBirdForTokenClassification
import pandas as pd 
import logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
service_folder = Path(__file__).parent.absolute()
from captum.attr import LayerIntegratedGradients

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

@dataclass
class SpliceAIConf:
    working_segment = 15000
    segment_step = 5000
    max_seq_len = 15000
    target_len = 15000
    max_tokens = 4096
    num_labels = 3
    tokenizer = service_folder.joinpath('data/tokenizers/t2t_1000h_multi_32k/')
    model_cfg = service_folder.joinpath('data/configs/hf_bigbird_L12-H768-A12-V32k-L4096.json')
    checkpoint_path = service_folder.joinpath('data/checkpoints/model_best.pth')
    attr_steps = 10
    reliable_pred_start: int = 5000
    reliable_pred_end: int = 10000

class SpliceaiService:
    ACCEPTORS_PRED_INDEX: int = 1
    DONORS_PRED_INDEX: int = 2

    def __init__(self, config: SpliceAIConf):
        self.conf = config
        # define model | labels: 0, 1, 2; multi-class multi-label classification
        model_cfg = AutoConfig.from_pretrained(config.model_cfg)
        model_cfg.num_labels = 3
        model_cfg.problem_type = 'multi_label_classification'
        model_cls = BigBirdForTokenClassification
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

        ## label counts in test set: [8378616.,    9842.,   10258.]) upweight class 1 and 2
        #self.pos_weight = torch.tensor([1.0, 100.0, 100.0])
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.tokenizer)
        self.wr_attr_count = 0

    def tokenize(self, seq):
        tokenized_seq = self.tokenizer(seq, 
                                  max_length=self.conf.max_tokens, 
                                  padding="max_length", 
                                  truncation="longest_first",
                                  return_tensors='pt')
        return tokenized_seq

    def __call__(self, query, temp_storage, calc_attr: bool = False) -> Dict:
        # model inference
        seq_tok = self.tokenize(query['seq'])

        with torch.inference_mode():
            predictions = torch.sigmoid(self.model(**seq_tok)["logits"]).detach().cpu().numpy().squeeze()

        # postprocessing
        service_response = dict()
        # write tokens
        # write predictions
        predictions = np.where(predictions > 0.5, 1, 0)

        service_response['acceptors'] = predictions[..., self.ACCEPTORS_PRED_INDEX] 
        service_response['donors'] = predictions[..., self.DONORS_PRED_INDEX] # [seq]
        service_response['tok_se'] = token_positions(seq_tok)
        service_response['query'] = query
        if calc_attr:
            attrs = self.annotate_predictions(sample=seq_tok, 
                                              preds=predictions,
                                              query=query,
                                              temp_storage=temp_storage)
            service_response['attrs'] = attrs

        return service_response
    
    def create_attr_object(self):
        def predict_classifier(inputs, 
                               token_type_ids=None, 
                               attention_mask=None):
            logger.info(f"{inputs.shape}")
            assert token_type_ids is not None
            assert attention_mask is not None
            output = self.model(inputs, 
                                token_type_ids=token_type_ids, 
                                attention_mask=attention_mask)
            return output.logits
        lig_object = LayerIntegratedGradients(predict_classifier, self.model.bert.embeddings)
        return lig_object
    
    def annotate_predictions(self, sample, preds, query, temp_storage):
        lig_object = self.create_attr_object()
        attributions = []    
        logger.info(f"Creating attributions for sample")


        smpl_attrs = {}
        tok2pos = token_positions(sample)
        
        lbord = max(query['lpad'], self.conf.reliable_pred_start)
        rbord = min(len(query['seq']) - query['rpad'], self.conf.reliable_pred_end)

        for ti in range(preds.shape[0]):
            raw_start, raw_end = tok2pos[ti]
            if raw_start >= raw_end:
                continue # special token

            if raw_end < lbord or raw_start > rbord:
                continue

            for pred_name, pred_ind in ( ('AC', self.ACCEPTORS_PRED_INDEX),
                                         ('DON', self.DONORS_PRED_INDEX)):
                if preds[ti, pred_ind] == 1:
                    attr = self.annotate_target(lig_object=lig_object, 
                                                sample=sample, 
                                                target=(ti, pred_ind), 
                                                query=query)
                    temp_path = temp_storage / f"attr_ind{ti}_target{pred_ind}_{self.wr_attr_count}.bed"
                    self.wr_attr_count += 1
                    attr.to_csv(temp_path, sep="\t", index=False)
                    smpl_attrs[(ti, pred_name)] = temp_path

        return smpl_attrs


    def annotate_target(self, lig_object, sample, target, query):

        lig_object = self.create_attr_object()

        attributions = lig_object.attribute(inputs=(sample['input_ids'],
                                            sample['token_type_ids'],
                                            sample['attention_mask']),
                                    target=target,
                                    n_steps=self.conf.attr_steps,
                                    return_convergence_delta=False)
        

        attributions = attributions[:, 1:-1, :] # remove CLS and SEP
        attributions = summarize_attributions(attributions).cpu()

        bed_like_table = {'tok_pos': [], 'token': [], 'attr': [], 'start': [], 'end': []}
        
        tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze(),
                                                         skip_special_tokens=False)
        startends = token_positions(sample)
        seq_len = len(query['seq'])
        for i, tok in enumerate(tokens):
           
            raw_start, raw_end = startends[i]
            
            if raw_start >= raw_end:
                continue # special token 
            logger.info(f"{raw_start} {raw_end} {query['rpad']} {query['lpad']} {tok}")

            if raw_end > seq_len - query['rpad']:
                # right padding, a not special token
                continue
            if raw_start < query['lpad']:
                # left padding, a not special token
                continue
            shift = query['context_start'] - query['lpad']
            start = raw_start + shift
            end = raw_end + shift 

            attr = attributions[i].item()
            bed_like_table['tok_pos'].append(i)
            bed_like_table['token'].append(tok)
            bed_like_table['attr'].append(attr)
           
            bed_like_table['start'].append(start)
            bed_like_table['end'].append(end)
        logger.info(f"{bed_like_table}")
        
        df = pd.DataFrame(bed_like_table)
                
        return df 