import requests


def main():
    url = "http://127.0.0.1:3000/splice_ai"
    # example =
    # # get | post | pull | delete
    # res = requests.post("http://127.0.0.1:3000/splice_ai", {"fasta_seq": example})
    #

    request_data = {
        "fasta_seq": ">chr1 \nCTCGTTCCGCGCCCGCCATGGAACCGGATGTACGTTATAGCTATTACGCTACTGTGGGTGCACTCGTTCCGCGCCCGCCATGGAACCGGATGGTCTAGCCGATCTGACGCTCGTTCCGCGCCCGCCATGGAACCGGATGCCCCGCCCCTGGTTTCGAGTCGCTGGCCTGCTGGGTGTCATCGCATTATCGATATTGCATTACGTTATAGCTATTACCTCGTTCCGCGCCCGCCATGGAACCGGATGGCTACTGTGGGTGCAGTCTAGC",
    }

    result = requests.post(url, json=request_data).json()
    print(result)

    # gold_result = [{"bad_words": True}, {"bad_words": False}, {"bad_words": True}]
    # assert result == gold_result, f"Got\n{result}\n, but expected:\n{gold_result}"
    # print("Success")


if __name__ == "__main__":
    main()
