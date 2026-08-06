"""Tests for CSV processor with complex protein ID format"""
import pytest
import tempfile
from pathlib import Path
from fridata.models.manage_dataset.csv_processor import CSVProcessor, ProteinEntry


class TestProteinEntry:
    """Test ProteinEntry dataclass"""
    
    def test_full_id_with_chain(self):
        """Test full_id property with chain"""
        entry = ProteinEntry(base_id="AF-A0A024B7W1-F1-model_v4", chain="8cxi_C")
        assert entry.full_id == "AF-A0A024B7W1-F1-model_v4__8cxi_C"
    
    def test_full_id_without_chain(self):
        """Test full_id property without chain"""
        entry = ProteinEntry(base_id="AF-A0A0A0K0K1-F1-model_v4")
        assert entry.full_id == "AF-A0A0A0K0K1-F1-model_v4"
    
    def test_full_id_with_nan_chain(self):
        """Test full_id property with 'nan' chain (should be treated as no chain)"""
        entry = ProteinEntry(base_id="AF-A0A0A0K0K1-F1-model_v4", chain="nan")
        assert entry.full_id == "AF-A0A0A0K0K1-F1-model_v4"
    
    def test_equality(self):
        """Test ProteinEntry equality"""
        entry1 = ProteinEntry(base_id="test", chain="A", ranges=[(1, 100)])
        entry2 = ProteinEntry(base_id="test", chain="A", ranges=[(1, 100)])
        entry3 = ProteinEntry(base_id="test", chain="B", ranges=[(1, 100)])
        
        assert entry1 == entry2
        assert entry1 != entry3
    
    def test_hash(self):
        """Test ProteinEntry hashing"""
        entry1 = ProteinEntry(base_id="test", chain="A", ranges=[(1, 100)])
        entry2 = ProteinEntry(base_id="test", chain="A", ranges=[(1, 100)])
        
        assert hash(entry1) == hash(entry2)
        assert len({entry1, entry2}) == 1  # Should deduplicate


class TestCSVProcessor:
    """Test CSVProcessor class"""
    
    def create_test_csv(self, content: str) -> Path:
        """Helper to create temporary CSV file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    def test_simple_csv_processing(self):
        """Test simple CSV format processing"""
        csv_content = """protein_id
AF-A0A0A0K0K1-F1-model_v4
AF-A0A024B7W1-F1-model_v4
"""
        csv_file = self.create_test_csv(csv_content)
        processor = CSVProcessor(csv_file)
        
        try:
            ids = processor.extract_ids()
            expected = ["AF-A0A0A0K0K1-F1-model_v4", "AF-A0A024B7W1-F1-model_v4"]
            assert ids == expected
        finally:
            csv_file.unlink()
    
    def test_complex_csv_processing(self):
        """Test complex CSV format processing with provided example"""
        csv_content = """AF-A0A0A0K0K1-F1-model_v4__nan,
AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403
AF-A0A024B7W1-F1-model_v4__8cxi_D,1-403
AF-A0A024B7W1-F1-model_v4__8cxi_E,202-613;700-712
"""
        csv_file = self.create_test_csv(csv_content)
        processor = CSVProcessor(csv_file)
        
        try:
            ids = processor.extract_complex_protein_ids()
            expected = [
                "AF-A0A0A0K0K1-F1-model_v4",
                "AF-A0A024B7W1-F1-model_v4__8cxi_C",
                "AF-A0A024B7W1-F1-model_v4__8cxi_E"
            ]
            assert ids == expected
        finally:
            csv_file.unlink()
    
    def test_range_deduplication(self):
        """Test that identical ranges are deduplicated"""
        csv_content = """AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403
AF-A0A024B7W1-F1-model_v4__8cxi_D,1-403
AF-A0A024B7W1-F1-model_v4__8cxi_E,202-613
"""
        csv_file = self.create_test_csv(csv_content)
        processor = CSVProcessor(csv_file)
        
        try:
            ids = processor.extract_complex_protein_ids()
            # Should only get C and E, D is deduplicated due to same range as C
            expected = [
                "AF-A0A024B7W1-F1-model_v4__8cxi_C",
                "AF-A0A024B7W1-F1-model_v4__8cxi_E"
            ]
            assert ids == expected
        finally:
            csv_file.unlink()
    
    def test_get_protein_entries(self):
        """Test getting full protein entries with range information"""
        csv_content = """AF-A0A0A0K0K1-F1-model_v4__nan,
AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403
AF-A0A024B7W1-F1-model_v4__8cxi_E,202-613;700-712
"""
        csv_file = self.create_test_csv(csv_content)
        processor = CSVProcessor(csv_file)
        
        try:
            entries = processor.get_protein_entries()
            assert len(entries) == 3
            
            # Check first entry (no ranges)
            assert entries[0].base_id == "AF-A0A0A0K0K1-F1-model_v4"
            assert entries[0].chain is None
            assert entries[0].ranges is None
            
            # Check second entry (single range)
            assert entries[1].base_id == "AF-A0A024B7W1-F1-model_v4"
            assert entries[1].chain == "8cxi_C"
            assert entries[1].ranges == [(1, 403)]
            
            # Check third entry (multiple ranges)
            assert entries[2].base_id == "AF-A0A024B7W1-F1-model_v4"
            assert entries[2].chain == "8cxi_E"
            assert entries[2].ranges == [(202, 613), (700, 712)]
            
        finally:
            csv_file.unlink()
    
    def test_parse_ranges(self):
        """Test range parsing functionality"""
        csv_file = self.create_test_csv("dummy,")
        processor = CSVProcessor(csv_file)
        
        try:
            # Single range
            ranges = processor._parse_ranges("1-403")
            assert ranges == [(1, 403)]
            
            # Multiple ranges
            ranges = processor._parse_ranges("202-613;700-712")
            assert ranges == [(202, 613), (700, 712)]
            
            # Range with spaces
            ranges = processor._parse_ranges(" 1 - 403 ; 700 - 712 ")
            assert ranges == [(1, 403), (700, 712)]
            
        finally:
            csv_file.unlink()
    
    def test_invalid_ranges(self):
        """Test error handling for invalid range specifications"""
        csv_file = self.create_test_csv("dummy,")
        processor = CSVProcessor(csv_file)
        
        try:
            # No dash
            with pytest.raises(ValueError, match="Range must contain '-'"):
                processor._parse_ranges("403")
            
            # Non-integer values
            with pytest.raises(ValueError, match="Range values must be integers"):
                processor._parse_ranges("a-403")
            
            # Start > end
            with pytest.raises(ValueError, match="Range start must be <= end"):
                processor._parse_ranges("403-1")
                
        finally:
            csv_file.unlink()
    
    def test_parse_complex_entry(self):
        """Test parsing individual complex entries"""
        csv_file = self.create_test_csv("dummy,")
        processor = CSVProcessor(csv_file)
        
        try:
            # Entry with chain and ranges
            entry = processor._parse_complex_entry("AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403")
            assert entry.base_id == "AF-A0A024B7W1-F1-model_v4"
            assert entry.chain == "8cxi_C"
            assert entry.ranges == [(1, 403)]
            
            # Entry with nan chain
            entry = processor._parse_complex_entry("AF-A0A0A0K0K1-F1-model_v4__nan,")
            assert entry.base_id == "AF-A0A0A0K0K1-F1-model_v4"
            assert entry.chain is None
            assert entry.ranges is None
            
            # Entry without chain
            entry = processor._parse_complex_entry("AF-A0A0A0K0K1-F1-model_v4,1-100")
            assert entry.base_id == "AF-A0A0A0K0K1-F1-model_v4"
            assert entry.chain is None
            assert entry.ranges == [(1, 100)]
            
        finally:
            csv_file.unlink()
    
    def test_missing_file(self):
        """Test error handling for missing CSV file"""
        processor = CSVProcessor(Path("/non/existent/file.csv"))
        
        with pytest.raises(FileNotFoundError):
            processor.extract_ids()
        
        with pytest.raises(FileNotFoundError):
            processor.extract_complex_protein_ids()
    
    def test_empty_csv(self):
        """Test handling of empty CSV file"""
        csv_file = self.create_test_csv("")
        processor = CSVProcessor(csv_file)
        
        try:
            ids = processor.extract_ids()
            assert ids == []
            
            ids = processor.extract_complex_protein_ids()
            assert ids == []
        finally:
            csv_file.unlink()
    
    def test_malformed_entries(self):
        """Test handling of malformed CSV entries"""
        csv_content = """AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403
invalid_entry_with_bad_range,bad-range
AF-A0A024B7W1-F1-model_v4__8cxi_E,202-613
"""
        csv_file = self.create_test_csv(csv_content)
        processor = CSVProcessor(csv_file)
        
        try:
            # Should skip malformed entries and continue processing
            ids = processor.extract_complex_protein_ids()
            expected = [
                "AF-A0A024B7W1-F1-model_v4__8cxi_C",
                "AF-A0A024B7W1-F1-model_v4__8cxi_E"
            ]
            assert ids == expected
        finally:
            csv_file.unlink()