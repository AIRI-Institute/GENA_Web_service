from datetime import date, datetime
from pathlib import Path

from server import save_fasta_and_faidx_files, save_annotations_files, instance_class

service_folder = Path(__file__).parent.absolute()


class SomeRequest:
    def __init__(self, input_obj):
        self.json = {'fasta_seq': input_obj}


if __name__ == "__main__":
    # define service instance
    # conf = PromotersConf()
    # instance_class = PromotersService(conf)

    # read test fasta file
    test_file_path = '/home/mks/airi_work/experiments/test_file/Large_pr_spl.fa'
    with open(test_file_path, 'r', encoding='utf-8') as fasta:
        fasta_content = fasta.read()

    # fasta file processing
    request = SomeRequest(input_obj=fasta_content)
    request_name = f"request_{date.today()}_{datetime.now().microsecond}"
    samples_queue, respond_dict = save_fasta_and_faidx_files(request, request_name)

    # run model on inputs sequences
    for sample_name, batches in samples_queue.items():
        sample_results = []
        for batch in batches[:3]:
            sample_results.append(instance_class(batch))  # Dicts with list 'seq'
            # and 'prediction' vector of batch size

        respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name)

    for key, val in respond_dict.items():
        print(key, val)
        print()
