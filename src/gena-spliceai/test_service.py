from datetime import date, datetime

from tqdm import tqdm

from server import save_fasta_and_faidx_files, save_annotations_files, instance_class, service_folder


class SomeForm:
    def __init__(self, input_obj):
        self.content = {'dna': input_obj}

    def get(self, name: str):
        return self.content[name]


class SomeRequest:
    # service_request.form.get('dna')
    def __init__(self, input_obj):
        self.form = SomeForm(input_obj)


if __name__ == "__main__":
    # read test fasta file
    # test_file = service_folder.joinpath('data/checkpoints/spliceAIsample_0.fa')
    test_file_path = service_folder.joinpath('data/checkpoints/Mid_spl.fa')
    with open(test_file_path, 'r', encoding='utf-8') as fasta:
        fasta_content = fasta.read()

    # fasta file processing
    # request = SomeRequest(input_obj='>chr1\nCTCGTTCCGCGCCCGCCATGGAACCGGATGTACGTTATAGCTATTACGCTACTGTGGGTGCACTCGTTCCGCGCCCGCCATGGAACCGGATGGTCTAGCCGATCTGACGCTCGTTCCGCGCCCGCCATGGAACCGGATGCCCCGCCCCTGGTTTCGAGTCGCTGGCCTGCTGGGTGTCATCGCATTATCGATATTGCATTACGTTATAGCTATTACCTCGTTCCGCGCCCGCCATGGAACCGGATGGCTACTGTGGGTGCAGTCTAGC')
    request = SomeRequest(input_obj=fasta_content)
    request_name = f"request_{date.today()}_{datetime.now().microsecond}"
    samples_queue, respond_dict = save_fasta_and_faidx_files(request, request_name)

    # run model on inputs sequences
    respond_dict['bed'] = []
    for sample_name, batches in samples_queue.items():
        sample_results = []
        for batch in tqdm(batches[:3]):
            sample_results.append(instance_class(batch))  # Dicts with list 'seq'
            # and 'prediction' vector of batch size

        respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name)

    # print output of the service
    for key, val in respond_dict.items():
        print(key, val)
        print()
