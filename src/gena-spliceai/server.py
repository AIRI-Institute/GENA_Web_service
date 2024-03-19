import os
import subprocess
import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple
import math
import numpy as np
import zipfile
from transformers import AutoModel, AutoTokenizer, AutoConfig, BigBirdForTokenClassification
import torch
import json
import textwrap
from Bio import SeqIO
import io
from pathlib import Path
import shutil 
from collections import defaultdict
import pandas as pd

from service import SpliceaiService, SpliceAIConf, service_folder

from pyfaidx import Faidx

from flask import Flask, request, jsonify
FASTA_CHARS_FOR_LINE = 100
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)
MAX_SEQ_SIZE: int = 10 ** 6
MAX_SEQ_IMPORTANCE_SIZE: int = 10 ** 4
def slice_sequence(seq: str, 
                   context_window: int,
                   pred_window: int,
                   max_tokens: int,
                   tokenizer,
                   resize_attempts: int = 10) -> list[str]:
    dv, m = divmod(context_window - pred_window, 2)
    stepL = dv
    stepR = dv + m

    slices = []

    for pred_start in range(0, len(seq), pred_window):
        pred_end = pred_start + pred_window
        if pred_end <= len(seq):
            addL = 0
            addR = 0
        else: # pred_end > len(seq)
            add_pad = pred_end - len(seq)
            pred_end = len(seq)
            dv, m = divmod(add_pad, 2)
            addL = dv
            addR = dv + m

        context_start = pred_start - stepL - addL
        context_end = pred_end + stepR + addR

        cur_slice = seq[pred_start:pred_end]
        if context_start >= 0:
            lpad = 0
            cur_slice = seq[context_start:pred_start] + cur_slice
        else: # context_start < 0
            lpad = -context_start
            context_start = 0
            cur_slice = 'N' * lpad + seq[context_start:pred_start] + cur_slice

        if context_end <= len(seq):
            rpad = 0 
            cur_slice = cur_slice + seq[pred_end:context_end]
        else: # context_end > len(seq)
            rpad = context_end - len(seq)
            context_end = len(seq)
            cur_slice = cur_slice + seq[pred_end:context_end] + 'N' * rpad

        tok = tokenizer.tokenize(cur_slice, add_special_tokens=True)
        for _ in range(resize_attempts):
            if len(tok) <= max_tokens:
                break
            
            if lpad != 0 or rpad != 0:
                lpad //= 2
                rpad //= 2

                if rpad != 0:
                    cur_slice = cur_slice[lpad:-rpad]
                else:
                    cur_slice = cur_slice[lpad:]
            else:
                left_shift = (pred_start - context_start) // 2
                right_shift = (context_end - pred_end) // 2
                if left_shift != 0 or right_shift != 0:
                    context_start += left_shift
                    context_end -= right_shift
                    cur_slice = cur_slice[left_shift:-right_shift]
                else:
                    break
            tok = tokenizer.tokenize(cur_slice, 
                                     add_special_tokens=True)
 
        slices.append({'seq': cur_slice,
                       'context_start': context_start,
                       'context_end': context_end,
                       'pred_start': pred_start,
                       'pred_end': pred_end,
                       'lpad': lpad,
                       'rpad': rpad})

    return slices

def processing_fasta_file(content: str, calc_importance: bool) -> Tuple:
    handle =  io.StringIO(content)

    records = SeqIO.parse(handle, format="fasta")

    file_queue = {}
    samples_content = {}
    sample_name = 'error'
    description = ''

    max_seq_size = MAX_SEQ_IMPORTANCE_SIZE if calc_importance else MAX_SEQ_SIZE

    for rec in records:
        sample_name = rec.name
        description = rec.description
        seq = str(rec.seq)
        assert len(seq) <= max_seq_size, 'Provided sequence is too large'
        file_queue[sample_name] = seq

        formatted_seq = "\n".join(textwrap.wrap(seq, width=FASTA_CHARS_FOR_LINE)) 
        if description:
            fileoneline = f">{sample_name} {description}\n{formatted_seq}\n"
        else:
            fileoneline = f">{sample_name}\n{formatted_seq}\n"
        samples_content[sample_name] = fileoneline

    return file_queue, samples_content, description


def save_fasta_and_faidx_files(fasta_content: str,
                                request_name: str, 
                                service: SpliceaiService,
                                calc_importance: bool) -> Tuple:
    faidx_time = time.time()

    respond_dict = {}
    samples_queue, samples_content, sample_desc = processing_fasta_file(fasta_content=fasta_content,
                                                                        calc_importance=calc_importance)
    for sample_name, dna_seq in samples_queue.items():
        st_time = time.time()

        # write fasta file
        file_name = f"{request_name}_{sample_name}"
        respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
        with respond_fa_file.open('w', encoding='utf-8') as fasta_file:
            fasta_file.write(samples_content[sample_name])

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_fasta_file"] = file_name + '.fa'
        respond_dict[f"fasta_file"] = file_name + '.fa'

        # splice dna sequence to necessary pieces
        samples_queue[sample_name] = slice_sequence(dna_seq, 
                    context_window=service.conf.working_segment,
                    pred_window=service.conf.segment_step,
                    max_tokens=service.conf.max_tokens,
                    tokenizer=service.tokenizer)
                
        total_time = time.time() - st_time
        logger.info(f"write {sample_name} fasta file exec time: {total_time:.3f}s")

        # write faidx file
        st_time = time.time()
        Faidx(respond_fa_file)

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_faidx_file"] = file_name + '.fa.fai'
        respond_dict[f"fai_file"] = file_name + '.fa.fai'

        total_time = time.time() - st_time
        logger.info(f"create and write {sample_name} faidx file exec time: {total_time:.3f}s")

    total_time = time.time() - faidx_time
    logger.info(f"create and write faidx file for all samples exec time: {total_time:.3f}s")

    return samples_queue, respond_dict, sample_desc
#  return dna_seq_names, req_path, all_tokenized_sequences, tokenizer


def get_model_prediction(batches, 
                         request_id,
                         service: SpliceaiService,
                         calc_importance: bool,
                         temp_storage: str):
    progress_file = f"data/respond_files/{request_id}_progress.json"
    cur_entries = 0
    total_entries = sum(map(len, batches))

    responses = []

    for ind, batch in enumerate(batches):
        logger.info(f"Processing entry {ind}")
        response = service(batch, 
                           temp_storage=temp_storage, 
                           calc_attr=calc_importance)
        responses.append(response)
        with open(progress_file, "w") as progress_fd:
            progress_fd.truncate(0)
            progress_fd.write(json.dumps({
                    "progress": math.ceil(cur_entries / total_entries * 100),
                    "cur_entries": cur_entries,
                    "total_entries": total_entries
            })
            )
        cur_entries += 1
    with open(progress_file, "w") as progress_fd:
            progress_fd.truncate(0)
            progress_fd.write(json.dumps({
                    "progress": math.ceil(cur_entries / total_entries * 100),
                    "cur_entries": cur_entries,
                    "total_entries": total_entries
            }))

    return responses


def save_annotations_files(seq_name, 
                           request_name,
                           responses):
    list_of_bed_files = []
 
    donors_name = f"{request_name}_{seq_name}_donors.bed"
    donors_respond_file = respond_files_path.joinpath(donors_name)

    acceptors_name = f"{request_name}_{seq_name}_acceptors.bed"
    acceptors_respond_file = respond_files_path.joinpath(acceptors_name)

    if not donors_respond_file.exists():
        with open(donors_respond_file, 'a', encoding='utf-8') as f_donors:
            print("track name=\"SD" + f"_{seq_name}" + "\"", file=f_donors)
    if not acceptors_respond_file.exists():
        with open(acceptors_respond_file, 'a', encoding='utf-8') as f_acceptors:
            print("track name=\"SA" + f"_{seq_name}" + "\"", file=f_acceptors)


    target_counts = defaultdict(int)
    with open(donors_respond_file, 'a', encoding='utf-8') as f_donors: # need / before results...
        with open(acceptors_respond_file, 'a', encoding='utf-8') as f_acceptors: # need / before results...
            for k in range(len(responses)):
                response = responses[k]
                acceptors = response['acceptors']
                donors = response['donors']
                tok_se = response['tok_se']
                query = response['query']
                seq_len = len(query['seq'])
                shift = query['context_start'] - query['lpad']
                attrs = response.get('attrs')
                logger.info(f"{query}")
                logger.info(f"{attrs}")
                for tok_pos in range(len(tok_se)):
                    raw_start, raw_end = tok_se[tok_pos]
                    if raw_start >= raw_end:
                        continue # special token 

                    if raw_end > seq_len - query['rpad']:
                        # right padding, a not special token
                        continue
                    if raw_start < query['lpad']:
                        # left padding, a not special token
                        continue
                   
                    start, end = raw_start + shift , raw_end + shift
                    if start < query['pred_start']:
                        continue
                    if end > query['pred_end']:
                        continue
                    
                    if donors[tok_pos] == 1:
                        target_counts['DON'] += 1
                        feature_name_numbered = f"SD_{target_counts['DON']}"
                        print(seq_name,
                              start, 
                              end,
                              feature_name_numbered, 
                              sep="\t", 
                              file=f_donors)
                        if attrs is not None:
                            attr_table = pd.read_table(attrs[(tok_pos, 'DON')])
                            attr_table['name'] = seq_name
                            attr_table = attr_table[['name', 'start', 'end', 'attr']]
                            attr_table_name = f"{request_name}_attributions_{feature_name_numbered}.bedGraph"
                            attr_table_path = respond_files_path.joinpath(attr_table_name)
                            attr_table.to_csv(attr_table_path, index=False, header=False, sep="\t")
                    if acceptors[tok_pos] == 1:
                        target_counts['AC'] += 1
                        feature_name_numbered = f"SA_{target_counts['AC']}"
                        print(seq_name, 
                              start, 
                              end,
                              feature_name_numbered, 
                              sep="\t",
                              file=f_acceptors)
                        if attrs is not None:
                            attr_table = pd.read_table(attrs[(tok_pos, 'AC')])
                            attr_table['name'] = seq_name
                            attr_table = attr_table[['name', 'start', 'end', 'attr']]
                            attr_table_name = f"{request_name}_attributions_{feature_name_numbered}.bedGraph"
                            attr_table_path = respond_files_path.joinpath(attr_table_name)
                            attr_table.to_csv(attr_table_path, index=False, header=False, sep="\t")
            
        list_of_bed_files.append(donors_respond_file) # need / before results...
        list_of_bed_files.append(acceptors_respond_file) # need / before results...

    bed_dict = {"bed": list_of_bed_files}

    return bed_dict


import threading
MAX_REQUESTS=5
sem = threading.Semaphore(value=MAX_REQUESTS)

@app.route("/api/gena-spliceai/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        request_id = request.form.get('id')
        assert request_id, 'Random id parameter required.'

        calc_importance = request.form.get('importance') == 'true'

        if 'file' in request.files:
            file = request.files['file']
            fasta_seq = file.read().decode('UTF-8')
        else:
            fasta_seq = request.form.get('dna')
        
        assert fasta_seq, 'Field DNA sequence or file are required.'

        try:
            request_name = f"request_{date.today()}_{datetime.now().microsecond}"
            with sem:
                conf = SpliceAIConf()
                service = SpliceaiService(conf)
                temp_storage_dir = Path(respond_files_path) / f"{request_id}_storage"
                temp_storage_dir.mkdir(exist_ok=True, parents=True)

                samples_queue, respond_dict, _ = save_fasta_and_faidx_files(fasta_content=fasta_seq,
                                                                            request_name=request_name,
                                                                            service=service,
                                                                            calc_importance=calc_importance)

                seq_name, batches = list(samples_queue.items())[0] # we use only first sequence from file
                responses = get_model_prediction(batches=batches, 
                                                 request_id=request_id,
                                                 service=service,
                                                 calc_importance=calc_importance,
                                                 temp_storage=temp_storage_dir)
                bed_dict = save_annotations_files(seq_name, 
                                                  request_name,
                                                  responses)
            shutil.rmtree(temp_storage_dir)

            archive_file_name = f"{request_name}_archive.zip"
            with zipfile.ZipFile(f"{respond_files_path}/{archive_file_name}", mode="w") as archive:
                archive.write(f"{respond_files_path}/{respond_dict['fasta_file']}", os.path.basename(respond_dict['fasta_file']))
                archive.write(f"{respond_files_path}/{respond_dict['fai_file']}", os.path.basename(respond_dict['fai_file']))
                for bed_file in bed_dict['bed']:
                    archive.write(bed_file, os.path.basename(bed_file))

            common_path = "/generated/gena-spliceai/"
            result = {
                "bed": [f"{common_path}/{Path(p).name}" for p in bed_dict["bed"]],
                "fasta_file": f"{common_path}{respond_dict['fasta_file']}",
                "fai_file": f"{common_path}{respond_dict['fai_file']}",
                "archive": f"{common_path}{archive_file_name}"
            }
            return jsonify(result)
        
        except AssertionError as e:
            return jsonify({'status':'error', 'message':str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
