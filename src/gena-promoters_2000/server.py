import logging
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Sized

from flask import Flask, request, jsonify
from pyfaidx import Faidx
import zipfile
import os

from service import service_folder, PromotersConf, PromotersService

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
conf = PromotersConf()
instance_class = PromotersService(conf)
respond_files_path = service_folder.joinpath('data/respond_files/')
respond_files_path.mkdir(exist_ok=True)


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


def processing_fasta_file(content: str) -> Tuple:
    file_queue = {}
    samples_content = {}
    sample_name = 'error'
    description = ''
    for line in content.splitlines():
        if line.startswith('>'):
            sample_name, description = processing_fasta_name(line)
            file_queue[sample_name] = ''
            samples_content[sample_name] = f">{sample_name} {description}\n"
        elif len(line) == 0:
            sample_name = 'error'
        else:
            file_queue[sample_name] += line
            samples_content[sample_name] += line + '\n'

    return file_queue, samples_content, description


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


def save_fasta_and_faidx_files(fasta_content: str, request_name: str) -> Tuple:
    faidx_time = time.time()

    respond_dict = {}
    samples_queue, samples_content, sample_desc = processing_fasta_file(fasta_content)
    for sample_name, dna_seq in samples_queue.items():
        st_time = time.time()

        # write fasta file
        file_name = f"{request_name}_{sample_name}"
        respond_fa_file = respond_files_path.joinpath(file_name + '.fa')
        with respond_fa_file.open('w', encoding='utf-8') as fasta_file:
            fasta_file.write(samples_content[sample_name])

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_fasta_file"] = file_name + '.fa'
        respond_dict[f"fasta_file"] = file_name + '.fa'

        # splice dna sequence to necessary pieces
        samples_queue[sample_name] = slicer(dna_seq, segment=conf.working_segment)  # , step=conf.segment_step
        samples_queue[sample_name] = slicer(samples_queue[sample_name], segment=conf.batch_size)  # List of batches

        total_time = time.time() - st_time
        logger.info(f"write {sample_name} fasta file exec time: {total_time:.3f}s")

        # write faidx file
        st_time = time.time()
        Faidx(respond_fa_file)

        # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
        #  с фронтом как обрабатывать такие случаи
        # respond_dict[f"{sample_name}_faidx_file"] = file_name + '.fa.fai'
        respond_dict[f"fai_file"] = file_name + '.fa.fai'

        total_time = time.time() - st_time
        logger.info(f"create and write {sample_name} faidx file exec time: {total_time:.3f}s")

    total_time = time.time() - faidx_time
    logger.info(f"create and write faidx file for all samples exec time: {total_time:.3f}s")

    return samples_queue, respond_dict, sample_desc


def save_annotations_files(annotation: List[Dict],
                           seq_name: str,
                           respond_dict: Dict,
                           request_name: str,
                           descriptions: str,
                           coding_type: str = 'utf-8',
                           delimiter: str = '\t') -> Dict:
    st_time = time.time()

    # create empty bed file
    file_name = f"{request_name}_{seq_name}_promoters.bed"
    respond_file = respond_files_path.joinpath(file_name)
    promoters_file = respond_file.open('w', encoding=coding_type)
    promoters_file.write(f'track name=promoters description="{descriptions}"\n')

    # add path to file in respond dict
    respond_dict['bed'].append(file_name)

    start = 0
    end = 0
    for batch_ans in annotation:
        for seq_element, prediction in zip(batch_ans['seq'], batch_ans['prediction']):
            if prediction == 1:
                end += conf.working_segment
            else:
                if (end != 0) and (start != end):
                    string = seq_name + delimiter + str(start) + delimiter + str(end) + '\n'  # delimiter + 'P'
                    promoters_file.write(string)
                    start = end
                else:
                    start += conf.working_segment
                    end += conf.working_segment

    if start != end:
        string = seq_name + delimiter + str(start) + delimiter + str(end) + '\n'  # delimiter + 'P'
        promoters_file.write(string)

    promoters_file.close()

    total_time = time.time() - st_time
    logger.info(f"write promoters bed files exec time: {total_time:.3f}s")

    return respond_dict


@app.route("/api/gena-promoters-2000/upload", methods=["POST"])
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
            samples_queue, respond_dict, descriptions = save_fasta_and_faidx_files(fasta_content, request_name)

            # run model on inputs sequences
            respond_dict['bed'] = []
            # todo: убрать заглушку на обработку только одной последовательности в fasta файле, после того договоримся
            #  с фронтом как обрабатывать такие случаи
            for sample_name, batches in list(samples_queue.items())[:1]:
                sample_results = []
                for batch in batches:
                    sample_results.append(instance_class(batch))  # Dicts with list 'seq'
                    # and 'prediction' vector of batch size

                respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name,
                                                      descriptions)

            # Генерируем архив
            archive_file_name = f"{request_name}_archive.zip"
            with zipfile.ZipFile(f"{respond_files_path}/{archive_file_name}", mode="w") as archive:
                archive.write(f"{respond_files_path}/{respond_dict['fasta_file']}", os.path.basename(respond_dict['fasta_file']))
                archive.write(f"{respond_files_path}/{respond_dict['fai_file']}", os.path.basename(respond_dict['fai_file']))
                for bed_file in respond_dict['bed']:
                    archive.write(f"{respond_files_path}/{bed_file}", os.path.basename(bed_file))

            # Генерируем url для файлов
            common_path = "/generated/gena-promoters-2000/"
            result = {
                "bed": [],
                "fasta_file": f"{common_path}{respond_dict['fasta_file']}",
                "fai_file": f"{common_path}{respond_dict['fai_file']}",
                "archive": f"{common_path}{archive_file_name}"
            }
            for bed_file_path in respond_dict['bed']:
               result['bed'].append(f"{common_path}{bed_file_path}")

            return jsonify(result)
        except AssertionError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
