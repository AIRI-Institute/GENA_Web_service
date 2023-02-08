import requests

from service import service_folder


def main():
    url = "http://127.0.0.1:3000/api/gena-promoters_2000/upload"

    # read test fasta file
    test_file_path = service_folder.joinpath('data/checkpoints/Mid_spl.fa')
    with open(test_file_path, 'r', encoding='utf-8') as fasta:
        fasta_content = fasta.read()

    request_data = {"fasta_seq": fasta_content}

    result = requests.post(url, json=request_data).json()
    print(result)


if __name__ == "__main__":
    main()
