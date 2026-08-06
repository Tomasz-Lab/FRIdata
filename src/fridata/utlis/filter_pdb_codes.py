from typing import List, Set, Iterable


def _read_pdb_types_(types: str) -> Set[str]:
    valid_types = {"prot", "prot-nuc"}

    allowed_pdbs_set = {
        parts[0]
        for line in types.strip().split("\n")
        if len(parts := line.strip().split()) >= 2 and parts[1] in valid_types
    }

    return allowed_pdbs_set


def filter_pdb_codes(pdb_mol_types: str, pdb_codes: Iterable[str]) -> List[str]:
    allowed_pdbs_set = _read_pdb_types_(pdb_mol_types)
    return list(set(pdb_codes) & allowed_pdbs_set)
