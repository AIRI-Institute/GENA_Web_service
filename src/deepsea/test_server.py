import requests


def main():
    url = "http://127.0.0.1:3000/deepsea"

    request_data = {
        "fasta_seq": ">chr1 \nTTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGATTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGATTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGATTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGATTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGATTTACTTTTAACATTTTGAAATATAAGACACCTAGAAAAAAGTTCACAGAAGGTAAATGTACACTTAAACAAATAAAGTGAGCACCCAAGTAGACACCACTTGAGCCAAGAACTGGAACATTACCAGCACCCCAGAAGCCCGATGGTATTCTTTCCCTATGGCAGCCCGTCTCAGAAAACAACCTTCCTCTCCCAGAAGA",
    }

    result = requests.post(url, json=request_data).json()
    print(result)


if __name__ == "__main__":
    main()
