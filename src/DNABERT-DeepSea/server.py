import os
import subprocess
import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from pyfaidx import Faidx

from flask import Flask, request, jsonify

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def save_fasta_and_faidx_files(service_request: request) -> Tuple[str, str, Dict]:
    st_time = time.time()
    req_path = f"/DNABERT_storage/request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}"
    os.mkdir(req_path)

    fasta_seq = service_request.form.get('dna')
    seq_name, dna_seq = fasta_seq.split('\n')
    chrome = seq_name.split()[0][1:]

    file_path = req_path + "/dev.csv"
    with open(file_path, 'w', encoding='utf-8') as input_file:
        i = 0
        pieces = []
        while (i < len(dna_seq)):
            piece = dna_seq[i:i+1024]
            pieces.append(piece)
            kmer = piece[:6]
            for j in range(1, len(piece) - 6 + 1):
                kmer += " " + piece[j:j+6]
            kmer += "\t0" * 919 + "\n"
            i += 1024
            input_file.write(kmer)

    file_path = req_path + "/dna.fa"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fasta_seq)

    Faidx(file_path)

    return pieces, chrome, req_path


def get_model_prediction(req_path: str):
    subprocess.run(["python3.9", "run_finetune.py", "--model_type", "dnalong", "--tokenizer_name=dna6", "--model_name_or_path", "/DNABERT6", "--task_name", "deepsea", "--do_predict", "--predict_dir", f"{req_path}", "--data_dir",  f"{req_path}", "--max_seq_length", "1024", "--per_gpu_pred_batch_size", "32", "--output_dir", "/DNABERT6", "--n_process",  "8"])


def save_annotations_files(pieces, chrome, req_path) -> Dict:

    annotation_table = pd.read_csv('annotation_table.csv', index_col='targetID')
    preds = np.load(req_path + '/pred_results.npy')
    json_files_bed = []

    # write bed files
    for file_type in annotation_table['FileName'].unique():
        json_files_bed.append(f"/generated/dnabert-deepsea{req_path}/result_{file_type}.bed")
        with open(req_path + f'/result_{file_type}.bed', 'w', encoding='utf-8') as f:

            f.write(f"track name=\"{file_type}\"\n")

            for i in range(len(pieces)):

                indexes = list(annotation_table[annotation_table['FileName'] == file_type].index)
                file_labels = preds[:, indexes]

                for n, batch_element in enumerate(file_labels):
                    for label, feature_index in zip(batch_element, indexes):
                        start = 200 * n
                        end = 200 * (n + 1)
                        if label == 1:
                            feature_name = annotation_table['RecordName'][feature_index]
                            f.write(f"{chrome}\t{str(start)}\t{str(end)}\t{feature_name}\n")

    return json_files_bed




@app.route("/api/dnabert-deepsea/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        pieces, chrome, req_path = save_fasta_and_faidx_files(request)
        get_model_prediction(req_path)
        bed_files = save_annotations_files(pieces, chrome, req_path)

        return jsonify({
            "bed": bed_files,
            "fasta_file":f"/generated/dnabert-deepsea{req_path}/dna.fa",
            "fai_file":f"/generated/dnabert-deepsea{req_path}/dna.fa.fai"
        })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)

