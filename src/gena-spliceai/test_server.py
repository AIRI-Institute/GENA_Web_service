import requests

from service import service_folder


def main():
    url = "http://127.0.0.1:3000/api/gena-spliceai/upload"

    test_file = service_folder.joinpath('data/checkpoints/Mid_spl.fa')
    with open(test_file, 'r') as tmp:
        file_text = tmp.read()

    request_data = {"fasta_seq": file_text}

    result = requests.post(url, json=request_data).json()
    print(result)


if __name__ == "__main__":
    main()
