from pathlib import Path

import requests

service_folder = Path(__file__).parent.absolute()


def main():
    url = "http://127.0.0.1:3000/gena-promoters"

    # read test fasta file
    test_file_path = service_folder.joinpath('data/checkpoints/Mid_spl.fa')
    with open(test_file_path, 'r', encoding='utf-8') as fasta:
        fasta_content = fasta.read()

    request_data = {"fasta_seq": fasta_content}

    result = requests.post(url, json=request_data).json()
    print(result)


if __name__ == "__main__":
    main()
