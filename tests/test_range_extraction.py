"""Tests for range-aware protein extraction"""
import pytest
from fridata.models.manage_dataset.sequences.sequence_and_coordinates_retriever import (
    __extract_sequences_and_coordinates__,
    _is_residue_in_ranges
)


class TestRangeFiltering:
    """Test range filtering functionality"""
    
    def test_is_residue_in_ranges_no_ranges(self):
        """Test that all residues pass when no ranges specified"""
        assert _is_residue_in_ranges(1, None) == True
        assert _is_residue_in_ranges(100, None) == True
        assert _is_residue_in_ranges(1000, None) == True
    
    def test_is_residue_in_ranges_single_range(self):
        """Test residue filtering with single range"""
        ranges = [(10, 20)]
        
        assert _is_residue_in_ranges(5, ranges) == False
        assert _is_residue_in_ranges(10, ranges) == True
        assert _is_residue_in_ranges(15, ranges) == True
        assert _is_residue_in_ranges(20, ranges) == True
        assert _is_residue_in_ranges(25, ranges) == False
    
    def test_is_residue_in_ranges_multiple_ranges(self):
        """Test residue filtering with multiple ranges"""
        ranges = [(10, 20), (30, 40), (50, 60)]
        
        assert _is_residue_in_ranges(5, ranges) == False
        assert _is_residue_in_ranges(15, ranges) == True
        assert _is_residue_in_ranges(25, ranges) == False
        assert _is_residue_in_ranges(35, ranges) == True
        assert _is_residue_in_ranges(45, ranges) == False
        assert _is_residue_in_ranges(55, ranges) == True
        assert _is_residue_in_ranges(65, ranges) == False
    
    def test_is_residue_in_ranges_empty_ranges(self):
        """Test residue filtering with empty ranges list"""
        ranges = []
        assert _is_residue_in_ranges(10, ranges) == False


class TestSequenceExtractionWithRanges:
    """Test sequence and coordinate extraction with range filtering"""
    
    def create_test_pdb_content(self):
        """Create test PDB content with multiple residues"""
        return """ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 50.00           C  
ATOM      2  CA  VAL A   2      11.000  21.000  31.000  1.00 50.00           C  
ATOM      3  CA  LEU A   3      12.000  22.000  32.000  1.00 50.00           C  
ATOM      4  CA  ILE A   4      13.000  23.000  33.000  1.00 50.00           C  
ATOM      5  CA  GLY A   5      14.000  24.000  34.000  1.00 50.00           C  
ATOM      6  CA  ALA A   6      15.000  25.000  35.000  1.00 50.00           C  
ATOM      7  CA  VAL A   7      16.000  26.000  36.000  1.00 50.00           C  
ATOM      8  CA  LEU A   8      17.000  27.000  37.000  1.00 50.00           C  
ATOM      9  CA  ILE A   9      18.000  28.000  38.000  1.00 50.00           C  
ATOM     10  CA  GLY A  10      19.000  29.000  39.000  1.00 50.00           C  
"""
    
    def test_extraction_without_ranges(self):
        """Test normal extraction without range filtering"""
        pdb_content = self.create_test_pdb_content()
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", None)
        
        # Should get full sequence
        assert sequence == "AVLIGAVLIG"
        assert len(coords) == 10
        
        # Check first and last coordinates
        assert coords[0] == (1, 10.0, 20.0, 30.0)
        assert coords[9] == (10, 19.0, 29.0, 39.0)
    
    def test_extraction_with_single_range(self):
        """Test extraction with single range filter"""
        pdb_content = self.create_test_pdb_content()
        ranges = [(3, 7)]  # Extract residues 3-7
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", ranges)
        
        # Should get partial sequence for residues 3-7
        assert sequence == "LIGAV"
        assert len(coords) == 5
        
        # Check coordinates match the range
        assert coords[0] == (3, 12.0, 22.0, 32.0)  # LEU at position 3
        assert coords[4] == (7, 16.0, 26.0, 36.0)  # VAL at position 7
    
    def test_extraction_with_multiple_ranges(self):
        """Test extraction with multiple range filters"""
        pdb_content = self.create_test_pdb_content()
        ranges = [(1, 3), (8, 10)]  # Extract residues 1-3 and 8-10
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", ranges)
        
        # Should get sequence from both ranges
        # Note: The function extracts from min to max, filling gaps with 'X'
        # So we get residues 1-10 with 4-7 filled with 'X'
        assert sequence == "AVLXXXXLIG"
        assert len(coords) == 10
        
        # Check that range 4-7 has None coordinates
        assert coords[0] == (1, 10.0, 20.0, 30.0)   # ALA
        assert coords[1] == (2, 11.0, 21.0, 31.0)   # VAL  
        assert coords[2] == (3, 12.0, 22.0, 32.0)   # LEU
        assert coords[3] == (4, None, None, None)    # Gap (filtered out)
        assert coords[4] == (5, None, None, None)    # Gap (filtered out)
        assert coords[5] == (6, None, None, None)    # Gap (filtered out)
        assert coords[6] == (7, None, None, None)    # Gap (filtered out)
        assert coords[7] == (8, 17.0, 27.0, 37.0)   # LEU
        assert coords[8] == (9, 18.0, 28.0, 38.0)   # ILE
        assert coords[9] == (10, 19.0, 29.0, 39.0)  # GLY
    
    def test_extraction_with_non_overlapping_ranges(self):
        """Test extraction with separate non-overlapping ranges"""
        pdb_content = self.create_test_pdb_content()
        ranges = [(2, 3), (7, 8)]  # Extract residues 2-3 and 7-8
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", ranges)
        
        # Should include residues 2-8 with gaps filled
        assert sequence == "VLXXXVL"
        assert len(coords) == 7  # From residue 2 to 8
        
        # Check specific coordinates
        assert coords[0] == (2, 11.0, 21.0, 31.0)   # VAL
        assert coords[1] == (3, 12.0, 22.0, 32.0)   # LEU
        assert coords[2] == (4, None, None, None)    # Gap
        assert coords[3] == (5, None, None, None)    # Gap
        assert coords[4] == (6, None, None, None)    # Gap
        assert coords[5] == (7, 16.0, 26.0, 36.0)   # VAL
        assert coords[6] == (8, 17.0, 27.0, 37.0)   # LEU
    
    def test_extraction_with_out_of_bounds_range(self):
        """Test extraction when range extends beyond available residues"""
        pdb_content = self.create_test_pdb_content()
        ranges = [(8, 15)]  # Range extends beyond residue 10
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", ranges)
        
        # Should only get residues 8-10 (what's available)
        assert sequence == "LIG"
        assert len(coords) == 3
        
        assert coords[0] == (8, 17.0, 27.0, 37.0)   # LEU
        assert coords[1] == (9, 18.0, 28.0, 38.0)   # ILE
        assert coords[2] == (10, 19.0, 29.0, 39.0)  # GLY
    
    def test_extraction_with_empty_range_result(self):
        """Test extraction when range doesn't match any residues"""
        pdb_content = self.create_test_pdb_content()
        ranges = [(20, 25)]  # Range completely outside available residues
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CA", ranges)
        
        # Should get empty results
        assert sequence == ""
        assert len(coords) == 0
    
    def test_extraction_with_cb_atoms_and_ranges(self):
        """Test range filtering with CB atoms"""
        pdb_content = """ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 50.00           C  
ATOM      2  CB  ALA A   1      10.100  20.100  30.100  1.00 50.00           C  
ATOM      3  CA  VAL A   2      11.000  21.000  31.000  1.00 50.00           C  
ATOM      4  CB  VAL A   2      11.100  21.100  31.100  1.00 50.00           C  
ATOM      5  CA  LEU A   3      12.000  22.000  32.000  1.00 50.00           C  
ATOM      6  CB  LEU A   3      12.100  22.100  32.100  1.00 50.00           C  
"""
        ranges = [(1, 2)]  # Extract only first two residues
        sequence, coords = __extract_sequences_and_coordinates__(pdb_content, "CB", ranges)
        
        assert sequence == "AV"
        assert len(coords) == 2
        
        # Should get CB coordinates, not CA
        assert coords[0] == (1, 10.1, 20.1, 30.1)   # ALA CB
        assert coords[1] == (2, 11.1, 21.1, 31.1)   # VAL CB