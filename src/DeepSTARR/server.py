import logging
import os
import subprocess
import time
from datetime import date, datetime
from typing import Dict, Tuple

import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def save_fasta_and_faidx_files(service_request: request) -> Tuple[str, str, Dict]:
    st_time = time.time()
    req_path = f"/DNABERT_storage/request_{date.today()}_{datetime.now().strftime('%H-%M-%S')}"
    os.mkdir(req_path)

    fasta_seq = service_request.json["fasta_seq"]
    seq_name, dna_seq = fasta_seq.split('\n')
    chrome = seq_name.split()[0][1:]

    file_path = req_path + "/dev.tsv"
    with open(file_path, 'w', encoding='utf-8') as input_file:
        input_file.write("seq\tDev_log2_enrichment\tHk_log2_enrichment\n")
        i = 0
        pieces = []
        while (i < len(dna_seq)):
            piece = dna_seq[i:i + 248]
            pieces.append(piece)
            kmer = piece[:6]
            for j in range(1, len(piece) - 6 + 1):
                kmer += " " + piece[j:j + 6]
            kmer += "\t0\t0\n"
            i += 248
            input_file.write(kmer)

    return pieces, chrome, req_path


def get_model_prediction(req_path: str):
    subprocess.run(
        ["python3.9", "run_finetune.py", "--model_type", "dna", "--tokenizer_name=dna6", "--model_name_or_path",
         "/DNABERT6", "--task_name", "dnaprom", "--do_predict", "--predict_dir", f"{req_path}", "--data_dir",
         f"{req_path}", "--max_seq_length", "243", "--per_gpu_eval_batch_size", "32", "--per_gpu_train_batch_size",
         "32", "--learning_rate", "2e-4", "--num_train_epochs", "5.0", "--output_dir", "/DNABERT6",
         "--evaluate_during_training", "--logging_steps", "100", "--save_steps", "100", "--warmup_percent", "0.1",
         "--hidden_dropout_prob", "0.1", "--overwrite_output", "--weight_decay", "0.01", "--n_process", "8"])


def save_annotations_files(pieces, chrome, req_path) -> Dict:
    with open(req_path + '/result.bed', 'w', encoding='utf-8') as f:
        preds = np.load(req_path + '/pred_results.npy')
        f.write("chrome\tstart\tend\tDev_log2_enrichment\tHk_log2_enrichment\n")
        for i in range(len(pieces)):
            f.write(f"{chrome}\t{str(i * 248)}\t{str((i + 1) * 248)}\t{str(preds[i, 0])}\t{str(preds[i, 1])}")


@app.route("/api/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        pieces, chrome, req_path = save_fasta_and_faidx_files(request)
        get_model_prediction(req_path)
        save_annotations_files(pieces, chrome, req_path)

        return jsonify({"path_to_bed_file": f"{req_path}/result.bed"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
