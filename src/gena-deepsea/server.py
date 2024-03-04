import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, Tuple, List, Sized, Optional
import gc

import numpy as np 
import pandas as pd
from flask import Flask, request, jsonify
from pyfaidx import Faidx
import zipfile
import os
import math
import json
from pathlib import Path
import shutil
from Bio import SeqIO
import io 
import textwrap

from service import DeepSeaConf, DeepSeaService, service_folder

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)
MAX_REQUESTS=5 
FASTA_CHARS_FOR_LINE = 100

def processing_fasta_file(content: str) -> Tuple:
    handle =  io.StringIO(content)

    records = SeqIO.parse(handle, format="fasta")

    file_queue = {}
    samples_content = {}
    sample_name = 'error'
    description = ''

    for rec in records:
        sample_name = rec.name
        description = rec.description
        seq = str(rec.seq)
        file_queue[sample_name] = seq

        formatted_seq = "\n".join(textwrap.wrap(seq, width=FASTA_CHARS_FOR_LINE)) 
        if description:
            fileoneline = f">{sample_name} {description}\n{formatted_seq}\n"
        else:
            fileoneline = f">{sample_name}\n{formatted_seq}\n"
        samples_content[sample_name] = fileoneline

    return file_queue, samples_content, description


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


def form_batches(seqs: list[str], batch_size: int):
    batches = []
    for s in range(0, len(seqs), batch_size):
        batches.append(seqs[s:s+batch_size])
    return batches


def save_fasta_and_faidx_files(fasta_content: str, request_name: str, service: DeepSeaService) -> Tuple:
    faidx_time = time.time()

    respond_dict = {}
    samples_queue, samples_content, sample_desc = processing_fasta_file(fasta_content)
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
        samples_queue[sample_name] = form_batches(
                slice_sequence(dna_seq, 
                    context_window=service.conf.working_segment,
                    pred_window=service.conf.segment_step,
                    max_tokens=service.conf.max_tokens,
                    tokenizer=service.tokenizer),
                batch_size=service.conf.batch_size)

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


def only_N(tok):
    return len(tok) == tok.count('N')


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           feature_counts: Dict[str, int],
                           request_name: str,
                           descriptions: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # read annotation file
    annotation_table = pd.read_csv(service_folder.joinpath('data/checkpoints/annotation_table.csv'),
                                   index_col='targetID')
    
    #feature_counts = defaultdict(int)
    # write bed files
    for file_type in annotation_table['FileName'].unique():
        # create bed file for labels group
        file_name = f"{request_name}_{seq_name}_{file_type}.bed"
        respond_file = respond_files_path.joinpath(file_name)
        
        if not respond_file.exists():
            with respond_file.open("w",  encoding=coding_type) as respond_file:
                print(f'track name={file_type} description="{descriptions}"\n', file=respond_file)

        with  respond_file.open('a', encoding=coding_type) as out_file :
            # add path to file in respond dict
            respond_dict['bed'].append(file_name)

            # get labels indices for the file type group
            indexes = list(annotation_table[annotation_table['FileName'] == file_type].index)
            # write info in bed file
            #n = 0


            for batch_ans in annotation:
                file_labels = batch_ans['prediction'][:, indexes] # [bs, indexes]
                
                attributions = batch_ans.get('attr')
                queries = batch_ans['queries']

                for be_ind, batch_element in enumerate(file_labels): # each batch element contains labels for each index
                    attr_dt = attributions[be_ind] if attributions is not None else None 
                    query = queries[be_ind]
                    for label, feature_index in zip(batch_element, indexes):
                        start = query['pred_start']
                        end = query['pred_end']
                        if label == 1:
                            feature_name = annotation_table['RecordName'][feature_index]
                            feature_counts[feature_name] += 1
                            feature_name_numbered = f"{feature_name}_{feature_counts[feature_name]}"
                            print(seq_name, start, end, feature_name_numbered, sep=delimiter, file=out_file)

                            if attr_dt is not None:

                                attr_table = pd.read_table(attr_dt[feature_index])
                                attr_table['name'] = seq_name
                                attr_table = attr_table[['name', 'start', 'end', 'attr']]
                            
                                attr_table_name = f"{request_name}_attributions_{feature_name_numbered}.bedGraph"
                                attr_table_path = respond_files_path.joinpath(attr_table_name)
                                #with open(attr_table_path, "w") as out: # written only once, no need for append mode
                                #   print(out, f'track name=bedGraph description="{feature_name}"')

                                attr_table.to_csv(attr_table_path, index=False, header=False, sep=delimiter)

    total_time = time.time() - st_time
    logger.info(f"write gena-deepsea bed files exec time: {total_time:.3f}s")

    return respond_dict, feature_counts

import threading
sem = threading.Semaphore(value=MAX_REQUESTS)

@app.route("/api/gena-deepsea/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        try:
            calc_importance = request.form.get('importance') == 'true'
            # create request unique name
            request_name = f"request_{date.today()}_{datetime.now().microsecond}"
            request_id = request.form.get('id')
            assert request_id, 'Random id parameter required.'

            # read data from request
            if 'file' in request.files:
                file = request.files['file']
                fasta_content = file.read().decode('UTF-8')
            else:
                fasta_content = request.form.get('dna')

            assert fasta_content, 'Field DNA sequence or file are required.'

            with sem:
                conf = DeepSeaConf()
                service = DeepSeaService(conf)

                samples_queue, respond_dict, descriptions = save_fasta_and_faidx_files(fasta_content, request_name, service)

                # run model on inputs sequences
                respond_dict['bed'] = []
                # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
                #  с фронтом как обрабатывать такие случаи
                progress_file = os.path.join(respond_files_path, f"{request_id}_progress.json")
                temp_storage_dir = Path(respond_files_path) / f"{request_id}_storage"
                temp_storage_dir.mkdir(exist_ok=True, parents=True)
                #logger.info(f"{samples_queue}")
                
                feature_counts = defaultdict(int)

                for sample_name, batches in list(samples_queue.items())[:1]:
                    cur_entries = 0
                    total_entries = sum(map(len, batches))
                    #sample_results = []
                    for batch in batches:
                        progress_fd = open(progress_file, "w")
                        progress_fd.truncate(0)
                        progress_fd.write(json.dumps({
                            "progress": math.ceil(cur_entries / total_entries * 100),
                            "cur_entries": cur_entries,
                            "total_entries": total_entries
                        }))
                        progress_fd.close()

                        #sample_results.append(service(batch))  # Dicts with list 'seq'
                        
                        answer = service(batch, temp_storage_dir, calc_importance)
                        respond_dict['bed'] = [] # temporary fix
                        respond_dict, feature_counts = save_annotations_files(annotation=[answer], # temp fix
                                                                seq_name=sample_name,
                                                                respond_dict=respond_dict,
                                                                feature_counts=feature_counts,
                                                                request_name=request_name,
                                                                descriptions=descriptions)

                        del answer 
                        gc.collect()


                        cur_entries += len(batch)
                del service
            shutil.rmtree(temp_storage_dir)
            #respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name, descriptions)
            
            # Генерируем архив
            archive_file_name = f"{request_name}_archive.zip"
            with zipfile.ZipFile(f"{respond_files_path}/{archive_file_name}", mode="w") as archive:
                archive.write(f"{respond_files_path}/{respond_dict['fasta_file']}", os.path.basename(respond_dict['fasta_file']))
                archive.write(f"{respond_files_path}/{respond_dict['fai_file']}", os.path.basename(respond_dict['fai_file']))
                for bed_file in respond_dict['bed']:
                    archive.write(f"{respond_files_path}/{bed_file}", os.path.basename(bed_file))

            # Генерируем url для файлов
            common_path = "/generated/gena-deepsea/"
            result = {
                "bed": [],
                "fasta_file": f"{common_path}{respond_dict['fasta_file']}",
                "fai_file": f"{common_path}{respond_dict['fai_file']}",
                "archive": f"{common_path}{archive_file_name}"
            }
            for bed_file_path in respond_dict['bed']:
                result['bed'].append(f"{common_path}{bed_file_path}")

            return jsonify(result)
        except AssertionError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, threaded=True) 
