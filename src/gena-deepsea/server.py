import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pyfaidx import Faidx

from src.service import DeepSeaConf, DeepSeaService, service_folder

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = DeepSeaConf()
instance_class = DeepSeaService(conf)
respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)


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

    # read annotation file
    annotation_table = pd.read_csv(service_folder.joinpath('data/checkpoints/annotation_table.csv'),
                                   index_col='targetID')

    # write bed files
    for file_type in annotation_table['FileName'].unique():
        file_name = f"request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}_{file_type}.bed"
        respond_file = respond_files_path.joinpath(file_name)
        respond_dict['acceptor_bed_file'] = str(respond_file)
        file = respond_file.open('w', encoding=coding_type)

        indexes = list(annotation_table[annotation_table['FileName'] == file_type].index)
        file_labels = annotation['prediction'][:, indexes]

        for n, batch_element in enumerate(file_labels):
            for label, feature_index in zip(batch_element, indexes):
                start = 200 * n
                end = 200 * (n + 1)
                if label == 1:
                    feature_name = annotation_table['RecordName'][feature_index]
                    string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + feature_name + '\n'
                    file.write(string)

        file.close()

    total_time = time.time() - st_time
    logger.info(f"write acceptor and donor bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/gena-deepsea", methods=["POST"])
def respond():
    if request.method == 'POST':
        dna_seq, chrome, respond_dict = save_fasta_and_faidx_files(request)
        model_out = get_model_prediction(dna_seq)
        result = save_annotations_files(model_out, chrome, respond_dict)

        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=3000)
