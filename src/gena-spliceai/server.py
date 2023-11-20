import os
import subprocess
import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple
import numpy as np
import zipfile
from transformers import AutoModel, AutoTokenizer, AutoConfig, BigBirdForTokenClassification
import torch

from pyfaidx import Faidx

from flask import Flask, request, jsonify

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def save_fasta_and_faidx_files(service_request: request) -> Tuple[str, str, Dict]:
    st_time = time.time()
    req_path = f"data/respond_files/request_{date.today()}_{datetime.now().microsecond}_" # without _
    os.makedirs(req_path)

    tokenizer = AutoTokenizer.from_pretrained('data/tokenizers/t2t_1000h_multi_32k/')

    # read data from request
    if 'file' in request.files:
        file = request.files['file']
        fasta_seq = file.read().decode('UTF-8')
    else:
        fasta_seq = request.form.get('dna')

    assert fasta_seq, 'Field DNA sequence or file are required.'
    # fasta_seq = service_request.json["fasta_seq"]
    # print(fasta_seq, flush=True)

    dna_seq_names = []
    dna_seqs = []
    flag = False
    for line in fasta_seq.splitlines():
        if line[0] == '>':
            if ':' in line:
                dna_seq_names.append(line.split(' ')[0].split(':')[0][1:])
            else:
                dna_seq_names.append(line.split(' ')[0][1:])
            flag = True
        else:
            if flag:
                dna_seqs.append('')
                flag = False
            dna_seqs[-1] += line

    all_tokenized_sequences = []
    for k in range(len(dna_seq_names)):
        all_tokenized_sequences.append([])
        dna_seq = dna_seqs[k]
        counter = 0
        while True:
            sub_dna_seq = dna_seq[counter:counter+15000]
            if len(sub_dna_seq) == 0:
                break
            else:
                tokenized_seq = tokenizer(sub_dna_seq, max_length=4096, padding="max_length", truncation="longest_first", return_tensors='pt')
                all_tokenized_sequences[k].append(tokenized_seq)  
                counter += 15000

    file_path = req_path + "dna.fa" # "/dna.fa"
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(len(dna_seq_names)):
            f.write('>' + dna_seq_names[i] + '\n')
            f.write(dna_seqs[i] + '\n')

    Faidx(file_path)

    return dna_seq_names, req_path, all_tokenized_sequences, tokenizer


def get_model_prediction(all_tokenized_sequences):
    # print('ok1', flush=True)
    model_cfg = AutoConfig.from_pretrained('data/configs/hf_bigbird_L12-H768-A12-V32k-L4096.json', num_labels=3)
    # print('ok2', flush=True)
    model = BigBirdForTokenClassification(model_cfg)
    # print('ok3', flush=True)
    model.load_state_dict(torch.load('data/checkpoints/model_best.pth', map_location='cpu')["model_state_dict"])
    # print('ok4', flush=True)
    model.eval()
    # print('ok5', flush=True)

    all_preds_donors = []
    all_preds_acceptors = []

    for k in range(len(all_tokenized_sequences)):
        all_preds_donors.append([])
        all_preds_acceptors.append([])
        tokenized_sequences_for_one_seq_name = all_tokenized_sequences[k]
        for i in range(len(tokenized_sequences_for_one_seq_name)):
            predictions = torch.sigmoid(model(**tokenized_sequences_for_one_seq_name[i])["logits"]).squeeze()
            # print(predictions.shape)
            all_preds_acceptors[-1] += list(predictions[:, 1].detach().cpu().numpy().squeeze())
            all_preds_donors[-1] += list(predictions[:, 2].detach().cpu().numpy().squeeze())

    print(all_preds_donors, flush=True)

    return all_preds_acceptors, all_preds_donors


def save_annotations_files(dna_seq_names, req_path, all_preds_acceptors, all_preds_donors, all_tokenized_sequences, tokenizer):

    global_counter = 0
    list_of_bed_files = []

    for j, seq_name in enumerate(dna_seq_names):
    
        with open(req_path + f'result_donors_{seq_name}.bed', 'w', encoding='utf-8') as fd: # need / before results...
            with open(req_path + f'result_acceptors_{seq_name}.bed', 'w', encoding='utf-8') as fa: # need / before results...
                all_preds_acceptors_mod = np.where(np.array(all_preds_acceptors[j]) > 0.5, 1, 0)
                all_preds_donors_mod = np.where(np.array(all_preds_donors[j]) > 0.5, 1, 0)

                # print(str(all_preds_acceptors_mod), flush=True)
                tokenized_sequences_for_one_seq_name = all_tokenized_sequences[j]

                # print(tokenized_sequences_for_one_seq_name[0]["input_ids"], flush=True)

                fd.write("track name=\"SD" + f"_{seq_name}" + "\"\n")
                fa.write("track name=\"SA" + f"_{seq_name}" + "\"\n")

                global_counter_bp = 0
                global_counter_tokens = 0
                for i in range(len(tokenized_sequences_for_one_seq_name)):
                    token_seq = tokenized_sequences_for_one_seq_name[i]["input_ids"][0]
                    tokens_bp_for_one_seq_name = tokenizer.convert_ids_to_tokens(token_seq)
                    # print(len(token_seq), flush=True)
                    for k in range(len(token_seq)):
                        if token_seq[k] not in [1, 2, 3]:
                            token_len_bp = len(tokens_bp_for_one_seq_name[k])
                            if all_preds_acceptors_mod[global_counter_tokens + k] == 1:
                                fa.write(f"{seq_name}\t{str(global_counter_bp)}\t{str(global_counter_bp + token_len_bp)}\t{str(1)}\n")
                            if all_preds_donors_mod[global_counter_tokens + k] == 1:
                                fd.write(f"{seq_name}\t{str(global_counter_bp)}\t{str(global_counter_bp + token_len_bp)}\t{str(1)}\n")
                            global_counter_bp += token_len_bp

                    global_counter_tokens += 4096

                
                # print(tokens_bp_for_one_seq_name, flush=True)

        list_of_bed_files.append(f"{req_path}donors_{seq_name}.bed") # need / before results...
        list_of_bed_files.append(f"{req_path}acceptors_{seq_name}.bed") # need / before results...

    bed_dict = {"bed": list_of_bed_files, "fasta_file":f"{req_path}dna.fa", "fai_file":f"{req_path}dna.fa.fai"} # need / before dna...

    return bed_dict


@app.route("/api/gena-spliceai/upload", methods=["POST"])
def respond():
    if request.method == 'POST':

        try:
            dna_seq_names, req_path, all_tokenized_sequences, tokenizer = save_fasta_and_faidx_files(request)
            all_preds_acceptors, all_preds_donors = get_model_prediction(all_tokenized_sequences)
            bed_dict = save_annotations_files(dna_seq_names, req_path, all_preds_acceptors, all_preds_donors, all_tokenized_sequences, tokenizer)

            archive_path = f"{req_path}archive.zip" # need / before archive...
            with zipfile.ZipFile(archive_path, mode="w") as archive:
                archive.write(bed_dict['fasta_file'], os.path.basename(bed_dict['fasta_file']))
                archive.write(bed_dict['fai_file'], os.path.basename(bed_dict['fai_file']))
                for bed_file in bed_dict['bed']:
                    archive.write(bed_file, os.path.basename(bed_file))

            common_path = "/generated/gena-spliceai/"
            result = {
                "bed": [],
                "fasta_file": f"{common_path}{bed_dict['fasta_file']}",
                "fai_file": f"{common_path}{bed_dict['fai_file']}",
                "archive": f"{common_path}{archive_path}"
            }
            for bed_file_path in bed_dict['bed']:
               result['bed'].append(f"{common_path}{bed_file_path}")

            return jsonify(result)
        
        except AssertionError as e:
            return jsonify({'status':'error', 'message':str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
