import logging
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Sized

import numpy as np
from flask import Flask, request, jsonify
from pyfaidx import Faidx

from service import service_folder, DeepStarrConf, DeepStarrService

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = DeepStarrConf()
instance_class = DeepStarrService(conf)
respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)


def processing_fasta_file(content: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    file_queue = {}
    samples_content = {}
    sample_name = 'error'
    for line in content.split('\n'):
        if line.startswith('>'):
            sample_name = line[1:]
            sample_name = sample_name.replace(' ', '_')
            sample_name = sample_name.replace("'", '_')
            sample_name = sample_name.replace('"', '_')

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
        elements.append(string[:-string_len])
    else:
        ind = 0
        while string_len >= segment:
            elements.append(string[(ind * segment):((ind + 1) * segment)])
            string_len -= segment
            ind += 1
        # добавляем оставшийся конец строки
        if string_len > 0:
            elements.append(string[:-string_len])

    return elements


def save_fasta_and_faidx_files(service_request: request, request_name: str) -> Tuple[Dict, Dict]:
    faidx_time = time.time()

    respond_dict = {}
    fasta_content = service_request.form.get('dna')
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
        # respond_dict[f"{sample_name}_fasta_file"] = '/generated/gena-deepstarr/' + file_name + '.fa'
        respond_dict[f"fasta_file"] = '/generated/gena-deepstarr/' + file_name + '.fa'

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
        # respond_dict[f"{sample_name}_faidx_file"] = '/generated/gena-deepstarr/' + file_name + '.fa.fai'
        respond_dict[f"fai_file"] = '/generated/gena-deepstarr/' + file_name + '.fa.fai'

        total_time = time.time() - st_time
        logger.info(f"create and write {sample_name} faidx file exec time: {total_time:.3f}s")

    total_time = time.time() - faidx_time
    logger.info(f"create and write faidx file for all samples exec time: {total_time:.3f}s")

    return samples_queue, respond_dict


def get_model_prediction(batch: List[str]) -> Dict:
    st_time = time.time()
    result = instance_class(batch)
    total_time = time.time() - st_time
    logger.info(f"gena-deepstarr model prediction exec time: {total_time:.3f}s")

    return result


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           request_name: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # # open fasta files for DEV
    dev_file_name = f"{request_name}_{seq_name}_dev.bed"
    respond_file = respond_files_path.joinpath(dev_file_name)
    dev_file = respond_file.open('w', encoding=coding_type)

    # open fasta files for HK
    hk_file_name = f"{request_name}_{seq_name}_hk.bed"
    respond_file = respond_files_path.joinpath(hk_file_name)
    hk_file = respond_file.open('w', encoding=coding_type)

    # add path to files in respond dict
    respond_dict['bed'].append('/generated/gena-deepstarr/' + dev_file_name)
    respond_dict['bed'].append('/generated/gena-deepstarr/' + hk_file_name)

    start = 0
    end = 0
    for batch_ans in annotation:
        for seq_element, dev_value, hk_value in zip(batch_ans['seq'], batch_ans['dev'], batch_ans['hk']):
            end += conf.working_segment
            # write dev results
            dev_string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + str(dev_value) + '\n'
            dev_file.write(dev_string)
            # write hk results
            hk_string = seq_name + delimiter + str(start) + delimiter + str(end) + delimiter + str(hk_value) + '\n'
            hk_file.write(hk_string)

            start += conf.working_segment

    dev_file.close()
    hk_file.close()

    total_time = time.time() - st_time
    logger.info(f"write dev and hk bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/api/gena-deepstarr/upload", methods=["POST"])
def respond():
    if request.method == 'POST':
        request_name = f"request_{date.today()}_{datetime.now().microsecond}"
        samples_queue, respond_dict = save_fasta_and_faidx_files(request, request_name)
        # run model on inputs sequences
        respond_dict['bed'] = []
        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        for sample_name, batches in list(samples_queue.items())[:1]:
            sample_results = []
            for batch in batches:
                sample_results.append(get_model_prediction(batch))  # Dicts with list 'seq'
                # and 'prediction' vector of batch size

            respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name)

        return jsonify(respond_dict)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)