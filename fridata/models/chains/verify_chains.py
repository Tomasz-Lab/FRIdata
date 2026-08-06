from typing import Dict



from fridata.models.manage_dataset.index.handle_index import read_index
from fridata.models.manage_dataset.structures_dataset import StructuresDataset
from fridata.models.manage_dataset.utils import read_all_pdbs_from_h5
from fridata.models.utils.cif2pdb import aa_dict

from fridata.utlis.logging import logger


def _parse_pdb_residue_(pdb_code, pdb_str):
    lines = pdb_str.split("\n")

    results = []
    for line in lines:
        # Check if the line starts with 'ATOM'
        if line.startswith("ATOM"):
            # Extract residue number (columns 23-26) and amino acid (columns 18-20)
            residue_number = int(line[22:26].strip())
            amino_acid = line[17:20].strip()

            try:
                short_acid = aa_dict[amino_acid]
                results.append((residue_number, short_acid))
            except KeyError:
                logger.warning(f"{pdb_code} {line} {amino_acid}")

    return dict(results)


def verify_chains(structures_dataset: StructuresDataset, pdb_seqres_fasta_path):
    proteins_index = read_index(
        structures_dataset.dataset_path() / "dataset_reversed.idx",
        structures_dataset.config.data_path
    )

    from Bio import SeqIO

    fasta_index = SeqIO.index(pdb_seqres_fasta_path, "fasta")

    results = []
    good_count = 0
    bad_count = 0

    for h5_file in proteins_index.keys():
        prots = read_all_pdbs_from_h5(h5_file)

        for p, content in prots.items():
            code = p.removesuffix(".pdb")
            acids_from_pdb: Dict[int, str] = _parse_pdb_residue_(code, content)

            try:
                seqres_sequence: str = fasta_index[code].seq
            except KeyError:
                logger.warning(f"The pdb_seqres index hasn't entry for the code {code}.")
                continue

            res: bool = _compare_from_pdb_vs_seqres_(
                code, acids_from_pdb, seqres_sequence, is_return_when_error=True
            )

            if res:
                good_count += 1
            else:
                bad_count += 1

            results.append(res)

    logger.info(f"Good results count: {good_count} ({float(good_count) / len(results)})")
    logger.info(f"Bad results count: {bad_count} ({float(bad_count) / len(results)})")


def _compare_from_pdb_vs_seqres_(
    code: str,
    from_pdb: Dict[int, str],
    from_seqres_str: str,
    is_return_when_error: bool = False,
):

    sequence_dict: Dict[int, str] = {
        i: char for i, char in enumerate(from_seqres_str, start=1)
    }
    is_all_good = True

    for key in from_pdb.keys():
        if from_pdb[key] != sequence_dict[key]:
            logger.warning(f"{code} {key} {from_pdb[key]} {sequence_dict[key]}")
            is_all_good = False
            if is_return_when_error:
                return False

    return is_all_good
