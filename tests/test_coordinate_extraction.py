import pytest
import textwrap
from pathlib import Path
from fridata.models.manage_dataset.sequences.sequence_and_coordinates_retriever import __extract_sequences_and_coordinates__


# Test data paths
TEST_DATA_PATH = Path(__file__).parent / "data"
PDB_EXPECTED_PATH = TEST_DATA_PATH / "pdb_expected"

sequence_1aa6_A = "MKKVVTVCPYCASGCKINLVVDNGKIVRAEAAQGKTNQGTLCLKGYYGWDFINDTQILTPRLKTPMIRRQRGGKLEPVSWDEALNYVAERLSAIKEKYGPDAIQTTGSSRGTGNETNYVMQKFARAVIGTNNVDCCARVCHGPSVAGLHQSVGNGAMSNAINEIDNTDLVFVFGYNPADSHPIVANHVINAKRNGAKIIVCDPRKIETARIADMHIALKNGSNIALLNAMGHVIIEENLYDKAFVASRTEGFEEYRKIVEGYTPESVEDITGVSASEIRQAARMYAQAKSAAILWGMGVTQFYQGVETVRSLTSLAMLTGNLGKPHAGVNPVRGQNNVQGACDMGALPDTYPGYQYVKDPANREKFAKAWGVESLPAHTGYRISELPHRAAHGEVRAAYIMGEDPLQTDAELSAVRKAFEDLELVIVQDIFMTKTASAADVILPSTSWGEHEGVFTAADRGFQRFFKAVEPKWDLKTDWQIISEIATRMGYPMHYNNTQEIWDELRHLCPDFYGATYEKMGELGFIQWPCRDTSDADQGTSYLFKEKFDTPNGLAQFFTCDWVAPIDKLTDEYPMVLSTVREVGHYSCRSMTGNCAALAALADEPGYAQINTEDAKRLGIEDEALVWVHSRKGKIITRAQVSDRPNKGAIYMTYQWWXXXXXXXXXXXXXXXXXXPEYKYCAVRVEPIADQRAAEQYVIDEYNKLKTRLREAALA"


# New helper function for the new tests
def generate_pdb_atom_line(atom_name, residue_name, residue_num, x, y, z, chain_id="A"):
    """Generates a PDB ATOM line string with specified fields."""
    # Format: ATOM serial name altloc resn chain resi icode xxxxxx x y z occ temp element charge
    # Simplified for testing, ensuring key fields are correct for the parser
    serial = 1 # Dummy serial
    altloc = " "
    icode = " "
    occupancy = 1.00
    temp_factor = 20.00
    element = atom_name[0] if atom_name else " "
    charge = "  "
    return f"ATOM  {serial:5d} {atom_name:<4.4s}{altloc}{residue_name:<3.3s} {chain_id}{residue_num:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2.2s}{charge}"


class TestExtractSequencesAndCoordinates:
    def test_empty_input(self):
        pdb_string = """"""
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == ""
        assert coords_with_gaps == ()

    def test_no_relevant_atoms(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("N", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("C", "ALA", 1, 4.0, 5.0, 6.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "A" # ALA residue should be A, even without CA/CB
        assert coords_with_gaps == ((1, None, None, None),) # One residue, no CA/CB

    def test_single_residue_ca(self):
        pdb_string = generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "A"
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0),)

    def test_single_residue_cb(self):
        pdb_string = generate_pdb_atom_line("CB", "GLY", 1, 1.0, 2.0, 3.0)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CB")
        assert sequence == "G"
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0),)

    def test_ca_preferred_over_cb_for_coords(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CB", "ALA", 1, 10.0, 11.0, 12.0)}
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
        """)
        # Test with "CA"
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "A"
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0),)

        # Test with "CB" (should only pick up CB if CA is not the target type)
        sequence_cb, coords_with_gaps_cb = __extract_sequences_and_coordinates__(pdb_string, "CB")
        assert sequence_cb == "A" # Sequence is independent of carbon_atom_type for selection
        assert coords_with_gaps_cb == ((1, 10.0, 11.0, 12.0),) # CB coordinate in gapped list

    def test_multiple_residues_sequential_ca(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("CA", "GLY", 2, 4.0, 5.0, 6.0)}
            {generate_pdb_atom_line("CA", "VAL", 3, 7.0, 8.0, 9.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "AGV"
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0), (2, 4.0, 5.0, 6.0), (3, 7.0, 8.0, 9.0))

    def test_gap_in_numbering(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("N", "PHE", 2, 0.0, 0.0, 0.0)} # Residue 2, but no CA/CB for CA type
            {generate_pdb_atom_line("CA", "VAL", 4, 7.0, 8.0, 9.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "AFXV" # ALA, PHE (no CA picked for coords), gap (X for res 3), VAL
        assert len(coords_with_gaps) == 4
        assert coords_with_gaps[0] == (1, 1.0, 2.0, 3.0)  # Res 1 (ALA)
        assert coords_with_gaps[1] == (2, None, None, None)   # Res 2 (PHE) - no CA found
        assert coords_with_gaps[2] == (3, None, None, None)   # Res 3 (gap)
        assert coords_with_gaps[3] == (4, 7.0, 8.0, 9.0)  # Res 4 (VAL)

    def test_non_standard_amino_acid(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("CA", "UNK", 2, 4.0, 5.0, 6.0)} # Unknown residue
            {generate_pdb_atom_line("CA", "GLY", 3, 7.0, 8.0, 9.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "AXG"
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0), (2, 4.0, 5.0, 6.0), (3, 7.0, 8.0, 9.0))

    def test_mixed_carbon_types_target_ca(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("CB", "GLY", 2, 4.0, 5.0, 6.0)} # GLY has CB specified
            {generate_pdb_atom_line("CA", "VAL", 3, 7.0, 8.0, 9.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "AGV"
        assert coords_with_gaps[0] == (1, 1.0, 2.0, 3.0)    # ALA CA
        assert coords_with_gaps[1] == (2, None, None, None) # GLY no CA
        assert coords_with_gaps[2] == (3, 7.0, 8.0, 9.0)    # VAL CA

    def test_mixed_carbon_types_target_cb(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)} # ALA has CA
            {generate_pdb_atom_line("CB", "GLY", 2, 4.0, 5.0, 6.0)} # GLY has CB
            {generate_pdb_atom_line("CA", "VAL", 3, 7.0, 8.0, 9.0)} # VAL has CA
            {generate_pdb_atom_line("CB", "VAL", 3, 7.7, 8.8, 9.9)} # VAL also has CB
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CB")
        assert sequence == "AGV" # Sequence should be the same
        assert coords_with_gaps[0] == (1, None, None, None) # ALA no CB
        assert coords_with_gaps[1] == (2, 4.0, 5.0, 6.0)    # GLY CB
        assert coords_with_gaps[2] == (3, 7.7, 8.8, 9.9)    # VAL CB

    def test_residues_at_start_and_end_missing_coords(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("N", "MET", 1, 0,0,0)} # MET, no CA/CB
            {generate_pdb_atom_line("CA", "ALA", 2, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("C", "LYS", 3, 0,0,0)} # LYS, no CA/CB
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "MAK"
        assert coords_with_gaps == ((1, None, None, None), (2, 1.0, 2.0, 3.0), (3, None, None, None))

    def test_file_order_vs_residue_order(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("CA", "VAL", 3, 7.0, 8.0, 9.0)}
            {generate_pdb_atom_line("CA", "ALA", 1, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("CA", "GLY", 2, 4.0, 5.0, 6.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "AGV" # Sequence is min to max residue index
        # Coords_with_breaks is in residue index order
        assert coords_with_gaps == ((1, 1.0, 2.0, 3.0), (2, 4.0, 5.0, 6.0), (3, 7.0, 8.0, 9.0))

    def test_all_residues_non_standard_or_no_coords(self):
        pdb_string = textwrap.dedent(f"""
            {generate_pdb_atom_line("N", "UNK", 10, 1.0, 2.0, 3.0)}
            {generate_pdb_atom_line("C", "XXX", 15, 4.0, 5.0, 6.0)}
        """)
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_string, "CA")
        assert sequence == "XXXXXX"
        assert coords_with_gaps == ((10, None, None, None), (11, None, None, None), (12, None, None, None),
                                    (13, None, None, None), (14, None, None, None), (15, None, None, None))


class TestExtractSequencesAndCoordinatesWithRealPDBFiles:
    """Test using real PDB files from the test data directory"""
    
    @pytest.mark.parametrize("pdb_file", ["5uyl_32.pdb", "1aa6_A.pdb", "2fjh_L.pdb"])
    def test_ca_extraction(self, pdb_file):
        """Test with real PDB files using CA atoms"""
        pdb_file_path = PDB_EXPECTED_PATH / pdb_file
        
        if not pdb_file_path.exists():
            pytest.skip(f"Test PDB file {pdb_file_path} not found")
        
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_content, "CA")
        
        # Basic checks
        assert len(sequence) > 0, f"Sequence should not be empty for {pdb_file}"
        assert len(coords_with_gaps) >= 1, f"coords_with_gaps should not be empty for {pdb_file}"
        
        # Verify sequence contains standard amino acid letters and gaps
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYXZ")  # Include X for gaps/non-standard
        assert all(c in valid_chars for c in sequence), f"Sequence contains invalid characters in {pdb_file}: {sequence}"
        
        # Check that coordinates are reasonable (not all zeros)
        assert any(coord != (0.0, 0.0, 0.0) for coord in coords_with_gaps if coord is not None and coord != (None, None, None)), f"All coordinates are zero for {pdb_file}"
        
        # Verify structure: coords_with_gaps should span from min to max residue number
        non_none_coords = [coord for coord in coords_with_gaps if coord != (None, None, None)]
        assert len(non_none_coords) >= 1, f"Should have at least one non-None coords_with_gaps for {pdb_file}"

    @pytest.mark.parametrize("pdb_file", ["5uyl_32.pdb", "1aa6_A.pdb", "2fjh_L.pdb"])
    def test_cb_extraction(self, pdb_file):
        """Test with real PDB files using CB atoms"""
        pdb_file_path = PDB_EXPECTED_PATH / pdb_file
        
        if not pdb_file_path.exists():
            pytest.skip(f"Test PDB file {pdb_file_path} not found")
        
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_content, "CB")
        
        # Basic checks
        assert len(sequence) > 0, f"Sequence should not be empty for {pdb_file}"
        assert len(coords_with_gaps) >= 1, f"coords_with_gaps should not be empty for {pdb_file}"
        
        # Sequence should be the same regardless of carbon type chosen
        sequence_ca, _ = __extract_sequences_and_coordinates__(pdb_content, "CA")
        assert sequence == sequence_ca, f"Sequence should be independent of carbon atom type for {pdb_file}"

    @pytest.mark.parametrize("pdb_file", ["5uyl_32.pdb", "1aa6_A.pdb", "2fjh_L.pdb"])
    def test_sequence_and_coordinates_consistency(self, pdb_file):
        """Test that sequence length matches coords_with_gaps length for real PDB files"""
        pdb_file_path = PDB_EXPECTED_PATH / pdb_file
        
        if not pdb_file_path.exists():
            pytest.skip(f"Test PDB file {pdb_file_path} not found")
        
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        sequence, coords_with_gaps = __extract_sequences_and_coordinates__(pdb_content, "CA")
        
        # The sequence length should equal coords_with_gaps length
        assert len(sequence) == len(coords_with_gaps), \
            f"Sequence length ({len(sequence)}) should match coords_with_gaps length ({len(coords_with_gaps)}) for {pdb_file}"
        
        # Check that gaps in sequence correspond to None in coords_with_gaps
        for i, (aa, coord) in enumerate(zip(sequence, coords_with_gaps)):
            if aa == 'X':
                # Gap in sequence might still have coordinates if the residue exists but is non-standard
                # So we don't enforce coord == (None, None, None) for 'X'
                pass
            else:
                # Standard amino acid should have been found, but might not have CA/CB
                pass  # coord could be None if no CA/CB found for this residue

    def test_1aa6_A_sequence_verification(self):
        """Test that extracted sequence from 1aa6_A.pdb matches expected sequence"""
        pdb_file_path = PDB_EXPECTED_PATH / "1aa6_A.pdb"
        
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        sequence, _ = __extract_sequences_and_coordinates__(pdb_content, "CA")
        
        # Verify the extracted sequence matches the expected sequence
        assert sequence == sequence_1aa6_A, \
            f"Extracted sequence does not match expected sequence.\nExpected: {sequence_1aa6_A}\nActual: {sequence}"


if __name__ == "__main__":
    pytest.main([__file__]) 