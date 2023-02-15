import os
import subprocess
import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple
import numpy as np

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

    file_path = req_path + "/dev.tsv"
    with open(file_path, 'w', encoding='utf-8') as input_file:
        input_file.write("seq\ttarget\n")
        counter_for_dna_seq_names = [0 for i in range(len(dna_seq_names))]
        for k in range(len(dna_seq_names)):
            dna_seq = dna_seqs[k]
            i = 0
            while (i < len(dna_seq)):
                piece = dna_seq[i:i+2048]
                kmer = piece[:6]
                for j in range(1, len(piece) - 6 + 1):
                    kmer += " " + piece[j:j+6]
                kmer += "\t0\n"
                i += 2048
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
    subprocess.run(["python3.9", "run_finetune.py", "--model_type", "dnalong", "--tokenizer_name=dna6", "--model_name_or_path", "/DNABERT6", "--task_name", "dnaprom", "--do_predict", "--predict_dir", f"{req_path}", "--data_dir",  f"{req_path}", "--max_seq_length", "2048", "--per_gpu_pred_batch_size", "32", "--output_dir", "/DNABERT6", "--n_process",  "8"])


def save_annotations_files(dna_seq_names, req_path, counter_for_dna_seq_names) -> Dict:

    global_counter = 0
    list_of_bed_files = []

    green_color = "\"255,51,51\""
    red_color = "\"153,255,51\""

    for j, seq_name in enumerate(dna_seq_names):
    
        with open(req_path + f'/result_{seq_name}.bed', 'w', encoding='utf-8') as f:
            preds = np.load(req_path + '/pred_results.npy')
            f.write("track name=\"Promoters" + f"_{seq_name}" + "\" visibility=2 itemRgb=On\n")
            for i in range(counter_for_dna_seq_names[j]):
                f.write(f"{seq_name}\t{str(i*2048)}\t{str((i+1)*2048)}\t{str(preds[global_counter])}\t{green_color if preds[global_counter] < 0.5 else red_color}\n")
                global_counter += 1

        list_of_bed_files.append(f"/generated/dnabert-promoters-2000{req_path}/result_{seq_name}.bed")

    bed_dict = {"bed": list_of_bed_files, "fasta_file":f"/generated/dnabert-promoters-2000{req_path}/dna.fa", "fai_file":f"/generated/dnabert-promoters-2000{req_path}/dna.fa.fai"}

    return bed_dict


@app.route("/api/dnabert-promoters-2000/upload", methods=["POST"])
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
