from datetime import date, datetime

from tqdm import tqdm

from server import save_annotations_files, instance_class, service_folder, processing_fasta_file, batch_reformat

if __name__ == "__main__":
    # read test fasta file
    test_file_path = service_folder.joinpath('data/checkpoints/FlyBase_XEGBPL.fasta')
    fasta = open(test_file_path, 'r', encoding='utf-8')

    # fasta file processing
    request_name = f"request_{date.today()}_{datetime.now().microsecond}"
    samples_queue, respond_dict, descriptions = processing_fasta_file(fasta, request_name)

    # run model on inputs sequences
    respond_dict["bed"] = []
    for sample_name, batches in list(samples_queue.items()):
        sample_results = []
        for batch in tqdm(batches):
            batch = batch_reformat(batch)
            sample_results.append(instance_class(batch))  # Dicts with list 'seq'
            # and 'prediction' vector of batch size
        respond_dict = save_annotations_files(sample_results, sample_name, respond_dict, request_name,
                                              descriptions[sample_name])

    # print output of the service
    for key, val in respond_dict.items():
        print(key, val)
        print()
