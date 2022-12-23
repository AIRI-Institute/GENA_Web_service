import requests


def main():
    url = "http://127.0.0.1:8922/splice_ai"  # 3000

    request_data = {
        "fasta_seq": ">chr1 \nCTCGTTCCGCGCCCGCCATGGAACCGGATGTACGTTATAGCTATTACGCTACTGTGGGTGCACTCGTTCCGCGCCCGCCATGGAACCGGATGGTCTAGCCGATCTGACGCTCGTTCCGCGCCCGCCATGGAACCGGATGCCCCGCCCCTGGTTTCGAGTCGCTGGCCTGCTGGGTGTCATCGCATTATCGATATTGCATTACGTTATAGCTATTACCTCGTTCCGCGCCCGCCATGGAACCGGATGGCTACTGTGGGTGCAGTCTAGC",
    }

    result = requests.post(url, json=request_data).json()
    print(result)


if __name__ == "__main__":
    main()
