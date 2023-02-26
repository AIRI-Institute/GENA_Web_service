import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple, Optional, Sized, List, TextIO

import numpy as np
from flask import Flask, request, jsonify
from pyfaidx import Faidx
from transformers import AutoTokenizer

from service import SpliceAIConf, SpliceaiService, service_folder

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = SpliceAIConf()
tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)
instance_class = SpliceaiService(conf)

respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)


def batch_reformat(batch: List):
    model_batch = {"input_ids": [],
                   "token_type_ids": [],
                   "attention_mask": []}

    max_seq_len = len(batch[0])

    for x in batch:
        token_type_ids = np.zeros(shape=max_seq_len, dtype=np.int64)
        attention_mask = np.array(x != tokenizer.pad_token_id, dtype=np.int64)

        model_batch['input_ids'].append(x)
        model_batch['token_type_ids'].append(token_type_ids)
        model_batch['attention_mask'].append(attention_mask)

    return model_batch


def slicer(string: Sized, segment: int, step: Optional[int] = None) -> List[str]:
    elements = list()
    string_len = len(string)

    if step is not None:
        ind = 0
        while string_len >= segment:
            elements.append(string[(ind * step):(ind * step) + segment])
            string_len -= step
            ind += 1
        # добавляем оставшийся конец строки
        elements.append(string[(ind * step):])
    else:
        ind = 0
        while string_len >= segment:
            elements.append(string[(ind * segment):((ind + 1) * segment)])
            string_len -= segment
            ind += 1
        # добавляем оставшийся конец строки
        if string_len > 0:
            elements.append(string[(ind * segment):])

    return elements


def processing_fasta_file(content: TextIO, request_name: str) -> Tuple[Dict, Dict]:
    tmp_dna = ''
    file_name = None
    tmp_file = None
    sample_name = None
    respond_fa_file = None

    file_queue = {}
    respond_dict = {}
    while True:
        line = content.readline()
        # if not line:  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
        #             #                  после того договоримся с фронтом как обрабатывать такие случаи
        #     break

        if line.startswith('>'):
            sample_name = line[1:].split()[0]
            if ':' in sample_name:
                sample_name = sample_name.split(':')[0]

            file_name = f"{request_name}_{sample_name}"
            respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
            tmp_file = open(respond_fa_file, 'w', encoding='utf-8')
            tmp_file.write(line)

        elif len(line) == 0:
            tmp_file.close()
            respond_dict[f"fasta_file"] = '/generated/gena-spliceai/' + file_name + '.fa'

            Faidx(respond_fa_file)
            respond_dict[f"fai_file"] = '/generated/gena-spliceai/' + file_name + '.fa.fai'

            # tokenization
            tmp_dna = tmp_dna.strip("N")
            mid_encoding = tokenizer(tmp_dna,
                                     add_special_tokens=True,
                                     padding="max_length",
                                     max_length=conf.max_seq_len,
                                     return_tensors="np")

            sample_seqs = slicer(mid_encoding["input_ids"][0], segment=conf.max_seq_len)
            file_queue[sample_name] = slicer(sample_seqs, segment=conf.batch_size)
            # tmp_dna = ''  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
            #                  после того договоримся с фронтом как обрабатывать такие случаи
            break

        else:
            tmp_file.write(line)
            tmp_dna += line

    # закрываем файл
    content.close()

    return file_queue, respond_dict


def processing_fasta_text(content: str, request_name: str):
    tmp_dna = ''
    file_name = None
    tmp_file = None
    sample_name = None
    respond_fa_file = None

    file_queue = {}
    respond_dict = {}
    lines = content.splitlines()
    for line in lines:
        if line.startswith('>'):
            sample_name = line[1:].split()[0]
            if ':' in sample_name:
                sample_name = sample_name.split(':')[0]

            file_name = f"{request_name}_{sample_name}"
            respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
            tmp_file = open(respond_fa_file, 'w', encoding='utf-8')
            tmp_file.write(line)

        elif len(line) == 0:
            tmp_file.close()
            respond_dict[f"fasta_file"] = '/generated/gena-spliceai/' + file_name + '.fa'

            Faidx(respond_fa_file)
            respond_dict[f"fai_file"] = '/generated/gena-spliceai/' + file_name + '.fa.fai'

            # tokenization
            tmp_dna = tmp_dna.strip("N")
            mid_encoding = tokenizer(tmp_dna,
                                     add_special_tokens=True,
                                     padding="max_length",
                                     max_length=conf.max_seq_len,
                                     return_tensors="np")

            sample_seqs = slicer(mid_encoding["input_ids"][0], segment=conf.max_seq_len)
            file_queue[sample_name] = slicer(sample_seqs, segment=conf.batch_size)
            # tmp_dna = ''  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
            #                  после того договоримся с фронтом как обрабатывать такие случаи
            break

        else:
            tmp_file.write(line)
            tmp_dna += line

    return file_queue, respond_dict


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           request_name: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # open fasta files for acceptors
    acceptor_file_name = f"{request_name}_{seq_name}_acceptors.bed"
    respond_acc_file = respond_files_path.joinpath(acceptor_file_name)
    acc_file = respond_acc_file.open('w', encoding=coding_type)
    acc_file.write(f'track name=SA description="GENA SpliceAI"\n')

    # open fasta files for donors
    donor_file_name = f"{request_name}_{seq_name}_donors.bed"
    respond_donor_file = respond_files_path.joinpath(donor_file_name)
    donor_file = respond_donor_file.open('w', encoding=coding_type)
    donor_file.write(f'track name=SD description="GENA SpliceAI"\n')

    # add paths to files in respond dict
    respond_dict['bed'].append('/generated/gena-spliceai/' + acceptor_file_name)
    respond_dict['bed'].append('/generated/gena-spliceai/' + donor_file_name)

    # chr start end (записи только для позитивного класса)
    start = 0
    end = 0
    for batch_ans in annotation:
        seq = tokenizer.convert_ids_to_tokens(batch_ans['input_ids'], skip_special_tokens=False)
        for token, acceptor, donor in zip(seq, batch_ans['acceptors'], batch_ans['donors']):
            if token not in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
                end += len(token)
                if acceptor == 1:
                    string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + token + '\n'
                    acc_file.write(string)

                elif donor == 1:
                    string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + token + '\n'
                    donor_file.write(string)

                start += len(token)

    acc_file.close()
    donor_file.close()

    total_time = time.time() - st_time
    logger.info(f"write acceptor and donor bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/api/gena-spliceai/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        try:
            # create request unique name
            request_name = f"request_{date.today()}_{datetime.now().microsecond}"

            # read data from request
            if 'file' in request.files:
                file = request.files['file']
                file = file.read().decode('UTF-8')
                samples_queue, respond_dict = processing_fasta_file(file, request_name)
            else:
                samples_queue, respond_dict = processing_fasta_text(request.form.get('dna'), request_name)

            assert samples_queue, 'Field DNA sequence or file are required.'

            # run model on inputs sequences
            respond_dict["bed"] = []
            for sample_name, batches in list(samples_queue.items()):
                sample_results = []
                for batch in batches:
                    batch = batch_reformat(batch)
                    sample_results.append(instance_class(batch))  # Dicts with list 'seq'
                    # and 'prediction' vector of batch size
                respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name)

            return jsonify(respond_dict)
        except AssertionError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
