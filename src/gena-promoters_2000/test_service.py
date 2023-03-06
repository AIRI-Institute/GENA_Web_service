from datetime import date, datetime

from tqdm import tqdm

from server import save_fasta_and_faidx_files, save_annotations_files, instance_class, service_folder

if __name__ == "__main__":
    # read test fasta file
    test_file_path = service_folder.joinpath('data/checkpoints/pr_spl_fwd_test.fa')
    with open(test_file_path, 'r', encoding='utf-8') as fasta:
        request = fasta.read()

    # fasta file processing
    request_name = f"request_{date.today()}_{datetime.now().microsecond}"
    samples_queue, respond_dict, description = save_fasta_and_faidx_files(request, request_name)

    # run model on inputs sequences
    respond_dict['bed'] = []
    for sample_name, batches in samples_queue.items():
        sample_results = []
        for batch in tqdm(batches):
            sample_results.append(instance_class(batch))  # Dicts with list 'seq'
            # and 'prediction' vector of batch size

        respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name, description)

    # print output of the service
    for key, val in respond_dict.items():
        print(key, val)
        print()
