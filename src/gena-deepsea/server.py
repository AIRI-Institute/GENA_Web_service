import logging
import time
from datetime import date, datetime
from typing import Dict, Tuple, List, Sized, Optional

import pandas as pd
from flask import Flask, request, jsonify
from pyfaidx import Faidx

from service import DeepSeaConf, DeepSeaService, service_folder

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = DeepSeaConf()
instance_class = DeepSeaService(conf)
respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)


def processing_fasta_file(content: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    file_queue = {}
    samples_content = {}
    sample_name = 'error'
    for line in content.splitlines():
        if line.startswith('>'):
            sample_name = line[1:]
            sample_name = sample_name.replace(' ', '_')
            sample_name = sample_name.replace("'", '_')

            file_queue[sample_name] = ''
            samples_content[sample_name] = line + '\n'
        elif len(line) == 0:
            sample_name = 'error'
        else:
            file_queue[sample_name] += line
            samples_content[sample_name] += line + '\n'

    return file_queue, samples_content


def slicer(string: Sized, segment: int, step: Optional[int] = None) -> List[str]:
    elements = list()
    string_len = len(string)
    if string_len < segment:
        string += 'N' * (segment - string_len)

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


def save_fasta_and_faidx_files(fasta_content: str, request_name: str) -> Tuple[Dict, Dict]:
    faidx_time = time.time()

    respond_dict = {}
    samples_queue, samples_content = processing_fasta_file(fasta_content)
    for sample_name, dna_seq in samples_queue.items():
        st_time = time.time()

        # write fasta file
        file_name = f"{request_name}_{sample_name}"
        respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
        with respond_fa_file.open('w', encoding='utf-8') as fasta_file:
            fasta_file.write(samples_content[sample_name])

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_fasta_file"] = '/generated/gena-promoters_2000/' + file_name + '.fa'
        respond_dict[f"fasta_file"] = '/generated/gena-deepsea/' + file_name + '.fa'

        # splice dna sequence to necessary pieces
        samples_queue[sample_name] = slicer(dna_seq, segment=conf.working_segment, step=conf.segment_step)
        samples_queue[sample_name] = slicer(samples_queue[sample_name], segment=conf.batch_size)  # List of batches

        total_time = time.time() - st_time
        logger.info(f"write {sample_name} fasta file exec time: {total_time:.3f}s")

        # write faidx file
        st_time = time.time()
        Faidx(respond_fa_file)

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_faidx_file"] = '/generated/gena-spliceai/' + file_name + '.fa.fai'
        respond_dict[f"fai_file"] = '/generated/gena-deepsea/' + file_name + '.fa.fai'

        total_time = time.time() - st_time
        logger.info(f"create and write {sample_name} faidx file exec time: {total_time:.3f}s")

    total_time = time.time() - faidx_time
    logger.info(f"create and write faidx file for all samples exec time: {total_time:.3f}s")

    return samples_queue, respond_dict


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           request_name: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # read annotation file
    annotation_table = pd.read_csv(service_folder.joinpath('data/checkpoints/annotation_table.csv'),
                                   index_col='targetID')

    # write bed files
    for file_type in annotation_table['FileName'].unique():
        # create bed file for labels group
        file_name = f"{request_name}_{seq_name}_{file_type}.bed"
        respond_file = respond_files_path.joinpath(file_name)
        file = respond_file.open('w', encoding=coding_type)
        file.write(f'track name={file_type} description="GENA chromatin annotation"\n')

        # add path to file in respond dict
        respond_dict['bed'].append('/generated/gena-deepsea/' + file_name)

        # get labels indices for the file type group
        indexes = list(annotation_table[annotation_table['FileName'] == file_type].index)
        # write info in bed file
        n = 0
        for batch_ans in annotation:
            file_labels = batch_ans['prediction'][:, indexes]

            for batch_element in file_labels:
                for label, feature_index in zip(batch_element, indexes):
                    start = conf.target_len * n
                    end = conf.target_len * (n + 1)
                    if label == 1:
                        feature_name = annotation_table['RecordName'][feature_index]
                        string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + feature_name
                        file.write(string + '\n')
                n += 1

        file.close()

    total_time = time.time() - st_time
    logger.info(f"write gena-deepsea bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/api/gena-deepsea/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        try:
            # create request unique name
            request_name = f"request_{date.today()}_{datetime.now().microsecond}"

            # read data from request
            if 'file' in request.files:
                file = request.files['file']
                fasta_content = file.read().decode('UTF-8')
            else:
                fasta_content = request.form.get('dna')

            assert fasta_content, 'Field DNA sequence or file are required.'

            # get queue of dna samples from fasta file
            samples_queue, respond_dict = save_fasta_and_faidx_files(fasta_content, request_name)

            # run model on inputs sequences
            respond_dict['bed'] = []
            # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
            #  с фронтом как обрабатывать такие случаи
            for sample_name, batches in list(samples_queue.items())[:1]:
                sample_results = []
                for batch in batches:
                    sample_results.append(instance_class(batch))  # Dicts with list 'seq'
                    # and 'prediction' vector of batch size

                respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name)

            return jsonify(respond_dict)
        except AssertionError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
