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
    req_path = f"/DNABERT_storage/request_{date.today()}_{datetime.now().microsecond}"
    os.mkdir(req_path)

    # read data from request
    if 'file' in request.files:
        file = request.files['file']
        fasta_seq = file.read().decode('UTF-8')
    else:
        fasta_seq = request.form.get('dna')

    assert fasta_seq, 'Field DNA sequence or file are required.'

    dna_seq_names = []
    dna_seqs = []
    flag = False
    for line in fasta_seq.splitlines():
        if line[0] == '>':
            dna_seq_names.append(line.split(' ')[0][1:])
            flag = True
        else:
            if flag:
                dna_seqs.append('')
                flag = False
            dna_seqs[-1] += line

    file_path = req_path + "/dev.csv"
    with open(file_path, 'w', encoding='utf-8') as input_file:
        counter_for_dna_seq_names = [0 for i in range(len(dna_seq_names))]
        for k in range(len(dna_seq_names)):
            dna_seq = dna_seqs[k]
            i = 0
            while (i < len(dna_seq)):
                piece = dna_seq[i:i+1024]
                kmer = piece[:6]
                for j in range(1, len(piece) - 6 + 1):
                    kmer += " " + piece[j:j+6]
                kmer += "\t0" * 919 + "\n"
                i += 1024
                input_file.write(kmer)
                counter_for_dna_seq_names[k] += 1

    file_path = req_path + "/dna.fa"
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(len(dna_seq_names)):
            f.write('>' + dna_seq_names[i] + '\n')
            f.write(dna_seqs[i] + '\n')

    Faidx(file_path)

    return dna_seq_names, req_path, counter_for_dna_seq_names


def get_model_prediction(req_path: str):
    subprocess.run(["python3.9", "run_finetune.py", "--model_type", "dnalong", "--tokenizer_name=dna6", "--model_name_or_path", "/DNABERT6", "--task_name", "deepsea", "--do_predict", "--predict_dir", f"{req_path}", "--data_dir",  f"{req_path}", "--max_seq_length", "1024", "--per_gpu_pred_batch_size", "32", "--output_dir", "/DNABERT6", "--n_process",  "8"])


def save_annotations_files(dna_seq_names, req_path, counter_for_dna_seq_names) -> Dict:

    annotation_table = pd.read_csv('annotation_table.csv', index_col='targetID')
    preds = np.where(np.load(req_path + '/pred_results.npy') > 0.5, 1, 0)
    list_of_bed_files = []
    global_counter = 0

    for j, seq_name in enumerate(dna_seq_names):

        # write bed files
        for file_type in annotation_table['FileName'].unique():
            list_of_bed_files.append(f"/generated/dnabert-deepsea{req_path}/result_{seq_name}_{file_type}.bed")
            with open(req_path + f'/result_{seq_name}_{file_type}.bed', 'w', encoding='utf-8') as f:

                f.write(f"track name=\"{file_type}" + f"_{seq_name}" + "\"\n")

                indexes = list(annotation_table[annotation_table['FileName'] == file_type].index)
                file_labels = preds[global_counter:global_counter + counter_for_dna_seq_names[j], indexes]

                for n, batch_element in enumerate(file_labels):
                    for label, feature_index in zip(batch_element, indexes):
                        start = 1024 * n
                        end = 1024 * (n + 1) # to-do: add proper length if input sequence length less than 1024 bp
                        if label == 1:
                            feature_name = annotation_table['RecordName'][feature_index]
                            f.write(f"{seq_name}\t{str(start)}\t{str(end)}\t{feature_name}\n")

        global_counter += counter_for_dna_seq_names[j]

    bed_dict = {"bed": list_of_bed_files, "fasta_file":f"/generated/dnabert-deepsea{req_path}/dna.fa", "fai_file":f"/generated/dnabert-deepsea{req_path}/dna.fa.fai"}

    return bed_dict




@app.route("/api/dnabert-deepsea/upload", methods=["POST"])
def respond():
    if request.method == 'POST':

        try: 
            dna_seq_names, req_path, counter_for_dna_seq_names = save_fasta_and_faidx_files(request)
            get_model_prediction(req_path)
            bed_dict = save_annotations_files(dna_seq_names, req_path, counter_for_dna_seq_names)

            return jsonify(bed_dict)

        except AssertionError as e:
            return jsonify({'status':'error', 'message':str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
