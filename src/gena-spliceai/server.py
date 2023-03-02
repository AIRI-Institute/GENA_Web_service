import logging
import os
import time
import zipfile
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


def sep_slicer(sized_obj: Sized, segment: int, step: Optional[int] = None, padding: bool = True) -> List[str]:
    segment = segment - 4
    left_edges = [tokenizer.convert_tokens_to_ids("[CLS]"),
                  tokenizer.convert_tokens_to_ids("[SEP]")]
    right_edges = [tokenizer.convert_tokens_to_ids("[SEP]"),
                   tokenizer.convert_tokens_to_ids("[SEP]")]

    elements = list()
    sized_obj_len = len(sized_obj)
    if step is not None:
        ind = 0
        while sized_obj_len >= segment:
            sub_seq = sized_obj[(ind * step):(ind * step) + segment]
            elements.append(left_edges + sub_seq + right_edges)
            sized_obj_len -= step
            ind += 1

        # добавляем оставшийся конец строки
        if padding:
            tail = sized_obj[(ind * step):]
            tail.extend([tokenizer.convert_tokens_to_ids("[PAD]")] * (segment - len(tail)))
            elements.append(left_edges + tail + right_edges)
        else:
            elements.append(sized_obj[(ind * step):])
    else:
        ind = 0
        while sized_obj_len >= segment:
            sub_seq = sized_obj[(ind * segment):((ind + 1) * segment)]
            elements.append(left_edges + sub_seq + right_edges)
            sized_obj_len -= segment
            ind += 1

        # добавляем оставшийся конец строки
        if sized_obj_len > 0:
            if padding:
                tail = sized_obj[(ind * segment):]
                tail.extend([tokenizer.convert_tokens_to_ids("[PAD]")] * (segment - len(tail)))
                elements.append(left_edges + tail + right_edges)
            else:
                elements.append(sized_obj[(ind * segment):])

    return elements


def batch_slicer(sized_obj: Sized, segment: int, step: Optional[int] = None) -> List[str]:
    elements = list()
    sized_obj_len = len(sized_obj)
    if step is not None:
        ind = 0
        while sized_obj_len >= segment:
            elements.append(sized_obj[(ind * step):(ind * step) + segment])
            sized_obj_len -= step
            ind += 1

        # добавляем оставшийся конец строки
        elements.append(sized_obj[(ind * step):])
    else:
        ind = 0
        while sized_obj_len >= segment:
            elements.append(sized_obj[(ind * segment):((ind + 1) * segment)])
            sized_obj_len -= segment
            ind += 1

        # добавляем оставшийся конец строки
        if sized_obj_len > 0:
            elements.append(sized_obj[(ind * segment):])

    return elements


def processing_fasta_name(desc_line: str) -> Tuple[str, str]:
    desc_line = desc_line[1:].strip()
    names = desc_line.split(' ')
    sample_name = names[0]
    if ':' in sample_name:
        seq_names = sample_name.split(':')
        sample_name = seq_names[0]
        description = seq_names[1]
    else:
        description = ' '.join(names[1:]).strip()

    return sample_name, description


def processing_fasta_file(content: TextIO, request_name: str) -> Tuple[Dict, Dict, Dict]:
    tmp_dna = ''
    file_name = None
    tmp_file = None
    sample_name = None
    respond_fa_file = None

    sample_desc = {}
    file_queue = {}
    respond_dict = {}
    while True:
        line = content.readline()
        # if not line:  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
        #             #                  после того договоримся с фронтом как обрабатывать такие случаи
        #     break

        if line.startswith('>'):
            sample_name, description = processing_fasta_name(line)

            file_name = f"{request_name}_{sample_name}"
            sample_desc[sample_name] = description
            respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
            tmp_file = open(respond_fa_file, 'w', encoding='utf-8')
            tmp_file.write(f">{sample_name} {description}\n")

        elif len(line) == 0:
            tmp_file.close()
            respond_dict[f"fasta_file"] = file_name + '.fa'

            Faidx(respond_fa_file)
            respond_dict[f"fai_file"] = file_name + '.fa.fai'

            # tokenization
            tmp_dna = tmp_dna.strip("N")
            dna_encoding = tokenizer(tmp_dna,
                                     add_special_tokens=False,
                                     padding=False,
                                     return_tensors="np")

            sample_seqs = sep_slicer(list(dna_encoding["input_ids"][0]), segment=conf.max_seq_len)
            file_queue[sample_name] = batch_slicer(sample_seqs, segment=conf.batch_size)
            # tmp_dna = ''  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
            #                  после того договоримся с фронтом как обрабатывать такие случаи
            break

        else:
            tmp_file.write(line)
            tmp_dna += line

    # закрываем файл
    content.close()

    return file_queue, respond_dict, sample_desc


def processing_fasta_text(content: str, request_name: str) -> Tuple[Dict, Dict, Dict]:
    tmp_dna = ''
    file_name = None
    tmp_file = None
    sample_name = None
    respond_fa_file = None

    sample_desc = {}
    file_queue = {}
    respond_dict = {}

    content += '\n\n'
    lines = content.splitlines()
    for line in lines:
        if line.startswith('>'):
            sample_name, description = processing_fasta_name(line)
            file_name = f"{request_name}_{sample_name}"
            sample_desc[sample_name] = description
            respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
            tmp_file = open(respond_fa_file, 'w', encoding='utf-8')
            tmp_file.write(f">{sample_name} {description}\n")

        elif len(line) == 0:
            tmp_file.close()
            respond_dict[f"fasta_file"] = file_name + '.fa'

            Faidx(respond_fa_file)
            respond_dict[f"fai_file"] = file_name + '.fa.fai'

            # tokenization
            tmp_dna = tmp_dna.strip("N")
            dna_encoding = tokenizer(tmp_dna,
                                     add_special_tokens=True,
                                     padding="max_length",
                                     max_length=conf.max_seq_len,
                                     return_tensors="np")

            sample_seqs = sep_slicer(list(dna_encoding["input_ids"][0]), segment=conf.max_seq_len)
            file_queue[sample_name] = batch_slicer(sample_seqs, segment=conf.batch_size)
            # tmp_dna = ''  # todo: убрать заглушку на обработку только одной последовательности в fasta файле,
            #                  после того договоримся с фронтом как обрабатывать такие случаи
            break

        else:
            tmp_file.write(line)
            tmp_dna += line

    return file_queue, respond_dict, sample_desc


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           request_name: str,
                           bed_descriptions: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # open fasta files for acceptors
    acceptor_file_name = f"{request_name}_{seq_name}_acceptors.bed"
    respond_acc_file = respond_files_path.joinpath(acceptor_file_name)
    acc_file = respond_acc_file.open('w', encoding=coding_type)
    acc_file.write(f'track name=SA description="{bed_descriptions}"\n')

    # open fasta files for donors
    donor_file_name = f"{request_name}_{seq_name}_donors.bed"
    respond_donor_file = respond_files_path.joinpath(donor_file_name)
    donor_file = respond_donor_file.open('w', encoding=coding_type)
    donor_file.write(f'track name=SD description="{bed_descriptions}"\n')

    # add paths to files in respond dict
    respond_dict['bed'].append(acceptor_file_name)
    respond_dict['bed'].append(donor_file_name)

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
                file.save(dst=str(respond_files_path.joinpath(f'{request_name}.fasta')))
                file = open(str(respond_files_path.joinpath(f'{request_name}.fasta')), 'r', encoding='utf-8')
                samples_queue, respond_dict, descriptions = processing_fasta_file(file, request_name)
            else:
                dna_seq = request.form.get('dna')
                samples_queue, respond_dict, descriptions = processing_fasta_text(dna_seq, request_name)

            assert samples_queue, 'Field DNA sequence or file are required.'

            # run model on inputs sequences
            respond_dict["bed"] = []
            for sample_name, batches in list(samples_queue.items()):
                sample_results = []
                for batch in batches:
                    batch = batch_reformat(batch)
                    sample_results.append(instance_class(batch))  # Dicts with list 'seq'
                    # and 'prediction' vector of batch size
                respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name,
                                                      descriptions[sample_name])

            # Генерируем архив
            archive_file_name = f"{request_name}_archive.zip"
            with zipfile.ZipFile(f"{respond_files_path}/{archive_file_name}", mode="w") as archive:
                archive.write(f"{respond_files_path}/{respond_dict['fasta_file']}",
                              os.path.basename(respond_dict['fasta_file']))
                archive.write(f"{respond_files_path}/{respond_dict['fai_file']}",
                              os.path.basename(respond_dict['fai_file']))
                for bed_file in respond_dict['bed']:
                    archive.write(f"{respond_files_path}/{bed_file}", os.path.basename(bed_file))

            # Генерируем url для файлов
            common_path = "/generated/gena-spliceai/"
            result = {"bed": [],
                      "fasta_file": f"{common_path}{respond_dict['fasta_file']}",
                      "fai_file": f"{common_path}{respond_dict['fai_file']}",
                      "archive": f"{common_path}{archive_file_name}"}
            for bed_file_path in respond_dict['bed']:
                result['bed'].append(f"{common_path}{bed_file_path}")

            return jsonify(result)
        except AssertionError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
