from collections import defaultdict

from Bio import SeqIO
import dask.bag as db


def read_fasta_as_generator(file_path):
    return SeqIO.parse(file_path, "fasta")


def process_record(record, smaller_set):
    if record.id in smaller_set:
        return (record.id, str(record.seq))
    return None


def check_sequence_consistency(smaller_seq, bigger_seq):
    for start in range(len(bigger_seq)):
        bigger_index = start
        smaller_index = 0
        while smaller_index < len(smaller_seq) and bigger_index < len(bigger_seq):
            if smaller_seq[smaller_index] == "-":
                smaller_index += 1
                bigger_index += 1
            elif smaller_seq[smaller_index] == bigger_seq[bigger_index]:
                smaller_index += 1
                bigger_index += 1
            else:
                print(smaller_index, smaller_seq[:smaller_index])
                print((start, bigger_index), bigger_seq[start:bigger_index])
                break
        if smaller_index == len(smaller_seq):
            return True  # Found a match
    return False  # No match found


def compare_fastas(smaller_fasta, bigger_fasta):
    # Read the smaller FASTA file and create a set of its protein IDs
    smaller_set = set(record.id for record in read_fasta_as_generator(smaller_fasta))

    # Create a Dask bag from the bigger FASTA file
    bigger_bag = db.from_sequence(
        read_fasta_as_generator(bigger_fasta), npartitions=100
    )

    # Process the bigger FASTA file and keep only the records present in the smaller set
    filtered_bag = bigger_bag.map(
        lambda record: process_record(record, smaller_set)
    ).filter(lambda x: x is not None)

    # Collect the results
    bigger_dict = dict(filtered_bag.compute())

    # Now process the smaller FASTA file and compare with the filtered bigger dict
    differences = []
    stats = defaultdict(int)

    for record in read_fasta_as_generator(smaller_fasta):
        protein_id = record.id
        smaller_seq = str(record.seq)

        if protein_id in bigger_dict:
            bigger_seq = bigger_dict[protein_id]
            if smaller_seq == bigger_seq:
                stats["same"] += 1
            else:
                stats["different"] += 1
                differences.append((protein_id, smaller_seq, bigger_seq))
        else:
            stats["missing"] += 1
            differences.append((protein_id, smaller_seq, "Not found in bigger FASTA"))

    # Print statistics
    total_proteins = len(smaller_set)

    from dask import delayed

    @delayed
    def process_diffs(protein_id, smaller_seq, bigger_seq):
        messages = []
        if bigger_seq != "Not found in bigger FASTA":
            # Check if smaller is shorter or equal in length to bigger
            if len(smaller_seq) > len(bigger_seq):
                messages.extend(
                    [
                        f"Warning: {protein_id} in smaller FASTA is longer than in bigger FASTA"
                    ]
                )

            # Check the consistency of smaller in bigger chain
            if not check_sequence_consistency(smaller_seq, bigger_seq):
                messages.extend(
                    [
                        f"Warning: {protein_id} sequence in smaller FASTA is not consistent with bigger FASTA"
                    ]
                )
                messages.extend([smaller_seq])
                messages.extend([bigger_seq])
                messages.extend(["---"])
        return messages

    process_diffs_tasks = [
        process_diffs(protein_id, smaller_seq, bigger_seq)
        for protein_id, smaller_seq, bigger_seq in differences
    ]
    process_diffs_results = db.from_delayed(process_diffs_tasks)

    for process_diff_result in process_diffs_results.compute():
        print(process_diff_result)

    # Print differences
    # print("\nDifferences found:")
    # for protein_id, seq1, seq2 in differences:
    #     print(f"Protein: {protein_id}")
    #     print(f"Smaller FASTA sequence: {seq1}")
    #     print(f"Bigger FASTA sequence: {seq2}")
    #     print("---")

    print(f"Total proteins in smaller FASTA: {total_proteins}")
    print(f"Identical proteins: {stats['same']}")
    print(f"Different proteins: {stats['different']}")
    print(f"Missing proteins: {stats['missing']}")
    print(
        f"Percentage of differences: {(stats['different'] + stats['missing']) / total_proteins * 100:.2f}%"
    )


if __name__ == "__main__":

    # smaller_fasta = "/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/datasets/PDB-subset--20240731_1535/sequences.fasta"
    # bigger_fasta = "/Users/youngdashu/downloads/pdb_seqres.txt"
    #
    # compare_fastas(smaller_fasta, bigger_fasta)

    small = "MVSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTL-VLCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKSNFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITHG"
    big = "MGSSHHHHHHSSGLVPRGSHMATMVSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLAYGVLCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKSNFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    # small = "BCD-AB--C"
    # big = "AABBCDABBACCC"

    res = check_sequence_consistency(small, big)

    print(res)
