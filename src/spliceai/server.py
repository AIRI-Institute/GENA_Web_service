import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple

import numpy as np
from flask import Flask, request, jsonify
from pyfaidx import Faidx

from src import service_folder
from src.service import SpliceAIConf, SpliceaiService

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = SpliceAIConf()
instance_class = SpliceaiService(conf)
respond_files_path = service_folder.joinpath('data/respond_files/spliceai')


def save_fasta_and_faidx_files(service_request: request) -> Tuple[str, str, Dict]:
    st_time = time.time()
    fasta_seq = service_request.json["fasta_seq"]
    seq_name, dna_seq = fasta_seq.split('\n')
    chrome = seq_name.split()[0][1:]

    # write fasta file
    file_name = f"request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}.fa"
    respond_fa_file = respond_files_path.joinpath(file_name)
    with respond_fa_file.open('w', encoding='utf-8') as fasta_file:
        fasta_file.write(fasta_seq)

    total_time = time.time() - st_time
    logger.info(f"write fasta file exec time: {total_time:.3f}s")

    # write faidx file
    st_time = time.time()

    Faidx(respond_fa_file)

    total_time = time.time() - st_time
    logger.info(f"create and write faidx file exec time: {total_time:.3f}s")

    return dna_seq, chrome, {'fasta_file': str(respond_fa_file), 'faidx_file': str(respond_fa_file) + '.fai'}


def get_model_prediction(dna_seq: str) -> np.array:
    st_time = time.time()
    result = instance_class(dna_seq)
    total_time = time.time() - st_time
    logger.info(f"splice_ai model prediction exec time: {total_time:.3f}s")

    return result


def save_annotations_files(annotation: Dict,
                           seq_name: str,
                           respond_dict: Dict,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # write fasta file
    acceptor_file_name = f"request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}_acceptor.bed"
    respond_acc_file = respond_files_path.joinpath(acceptor_file_name)
    respond_dict['acceptor_bed_file'] = str(respond_acc_file)
    acc_file = respond_acc_file.open('w', encoding=coding_type)

    donor_file_name = f"request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}_donor.bed"
    respond_donor_file = respond_files_path.joinpath(donor_file_name)
    respond_dict['donor_bed_file'] = str(respond_donor_file)
    donor_file = respond_donor_file.open('w', encoding=coding_type)

    # chr start end (записи только для позитивного класса)
    start = 0
    end = 0
    for token, label in zip(annotation['seq'], annotation['prediction']):
        if token not in ['[CLS]', '[SEP]', '[UNK]']:
            end += len(token)
            if label == 1:
                string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + token + '\n'
                acc_file.write(string)
            elif label == 2:
                string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + token + '\n'
                donor_file.write(string)

            start += len(token)

    acc_file.close()
    donor_file.close()

    total_time = time.time() - st_time
    logger.info(f"write acceptor and donor bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/splice_ai", methods=["POST"])
def respond():
    if request.method == 'POST':
        dna_seq, chrome, respond_dict = save_fasta_and_faidx_files(request)
        model_out = get_model_prediction(dna_seq)
        result = save_annotations_files(model_out, chrome, respond_dict)

        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=3000)