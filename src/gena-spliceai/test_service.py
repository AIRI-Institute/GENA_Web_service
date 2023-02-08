from server import instance_class, service_folder

if __name__ == "__main__":
    test_file = service_folder.joinpath('data/checkpoints/spliceAIsample_0.fa')
    with open(test_file, 'r') as tmp:
        lines = tmp.readlines()
        chrome_name, dna_seq = lines[0], lines[1]

    y = instance_class(dna_seq)

    print(f"Acceptors: {y['acceptors']}")
    print(f"Donors: {y['donors']}")
    print(y['seq'])
