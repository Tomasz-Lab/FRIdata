"""CSV processor for handling CSV ID files"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Set


@dataclass
class ProteinEntry:
    """Represents a protein entry with ID, chain, and range specifications"""
    base_id: str
    chain: Optional[str] = None
    ranges: Optional[List[Tuple[int, int]]] = None
    
    @property
    def full_id(self) -> str:
        """Get the full protein ID (base_id + chain if present)"""
        if self.chain and self.chain != 'nan':
            return f"{self.base_id}__{self.chain}"
        return self.base_id
    
    def __hash__(self):
        return hash((self.base_id, self.chain, tuple(self.ranges) if self.ranges else None))
    
    def __eq__(self, other):
        if not isinstance(other, ProteinEntry):
            return False
        return (self.base_id, self.chain, self.ranges) == (other.base_id, other.chain, other.ranges)


class CSVProcessor:
    """Handles processing of CSV files containing protein IDs"""
    
    def __init__(self, file_path: Path):
        """Initialize CSV processor with file path
        
        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        
    def extract_ids(self, column_index: int = 0, has_header: bool = True) -> List[str]:
        """Extract protein IDs from CSV file
        
        Args:
            column_index: Column index containing the IDs (default: 0)
            has_header: Whether CSV file has header row (default: True)
            
        Returns:
            List of protein IDs
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If column index is invalid
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file {self.file_path} does not exist")
            
        ids = []
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header if present
            if has_header:
                next(csv_reader, None)
                
            for row in csv_reader:
                if len(row) > column_index:
                    protein_id = row[column_index].strip()
                    if protein_id:  # Only add non-empty IDs
                        ids.append(protein_id)
                else:
                    raise ValueError(f"Column index {column_index} not found in row: {row}")
                    
        return ids
    
    def extract_complex_protein_ids(self, column_index: int = 0, has_header: bool = False) -> List[str]:
        """Extract protein IDs from CSV file with complex format handling
        
        Handles format: protein_id[__chain],range1-range2;range3-range4
        Example: AF-A0A024B7W1-F1-model_v4__8cxi_C,1-403
        
        Args:
            column_index: Column index containing the IDs (default: 0)
            has_header: Whether CSV file has header row (default: False)
            
        Returns:
            List of processed protein IDs
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If column index is invalid
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file {self.file_path} does not exist")
            
        entries = []
        seen_ranges: dict[tuple, str] = {}  # Track ranges to IDs for deduplication
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header if present
            if has_header:
                next(csv_reader, None)
                
            for row_idx, row in enumerate(csv_reader):
                if len(row) <= column_index:
                    continue
                    
                # Reconstruct the original format: protein_id,ranges
                protein_id = row[column_index].strip()
                if not protein_id:
                    continue
                    
                # Get range part (if exists)
                range_part = ""
                if len(row) > column_index + 1:
                    range_part = row[column_index + 1].strip()
                
                # Reconstruct the full entry format
                if range_part:
                    raw_entry = f"{protein_id},{range_part}"
                else:
                    raw_entry = f"{protein_id},"
                    
                try:
                    entry = self._parse_complex_entry(raw_entry)
                    if entry:
                        # Check for range deduplication
                        if entry.ranges:
                            range_key = tuple(sorted(entry.ranges))
                            if range_key in seen_ranges:
                                # Skip duplicate range, keep first occurrence
                                continue
                            seen_ranges[range_key] = entry.full_id
                        
                        entries.append(entry)
                        
                except Exception as e:
                    # Log warning but continue processing
                    print(f"Warning: Error parsing row {row_idx + 1}: {raw_entry} - {e}")
                    continue
        
        return [entry.full_id for entry in entries]
    
    def _parse_complex_entry(self, raw_entry: str) -> Optional[ProteinEntry]:
        """Parse a single complex CSV entry
        
        Format: protein_id[__chain],range1-range2;range3-range4
        
        Args:
            raw_entry: Raw CSV entry string
            
        Returns:
            ProteinEntry object or None if parsing fails
        """
        # Split on comma to separate ID from ranges
        parts = raw_entry.split(',', 1)
        id_part = parts[0].strip()
        range_part = parts[1].strip() if len(parts) > 1 else None
        
        # Parse ID and chain
        if '__' in id_part:
            # Complex format with chain
            base_id, chain_part = id_part.split('__', 1)
            chain = chain_part if chain_part != 'nan' else None
        else:
            # Simple format
            base_id = id_part
            chain = None
            
        # Parse ranges if present
        ranges = None
        if range_part and range_part.strip():
            try:
                ranges = self._parse_ranges(range_part)
            except ValueError as e:
                raise ValueError(f"Invalid range specification '{range_part}': {e}")
        
        return ProteinEntry(base_id=base_id.strip(), chain=chain, ranges=ranges)
    
    def _parse_ranges(self, range_str: str) -> List[Tuple[int, int]]:
        """Parse range specifications
        
        Args:
            range_str: Range string like "1-403" or "202-613;700-712"
            
        Returns:
            List of (start, end) tuples
            
        Raises:
            ValueError: If range format is invalid
        """
        ranges = []
        
        # Split by semicolon for multiple ranges
        for range_part in range_str.split(';'):
            range_part = range_part.strip()
            if not range_part:
                continue
                
            # Split by dash for start-end
            if '-' not in range_part:
                raise ValueError(f"Range must contain '-': {range_part}")
                
            start_str, end_str = range_part.split('-', 1)
            
            try:
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError:
                raise ValueError(f"Range values must be integers: {range_part}")
                
            if start > end:
                raise ValueError(f"Range start must be <= end: {range_part}")
                
            ranges.append((start, end))
            
        return ranges
    
    def get_protein_entries(self, column_index: int = 0, has_header: bool = False) -> List[ProteinEntry]:
        """Get full protein entries with range information for downstream processing
        
        Args:
            column_index: Column index containing the IDs (default: 0)
            has_header: Whether CSV file has header row (default: False)
            
        Returns:
            List of ProteinEntry objects
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file {self.file_path} does not exist")
            
        entries = []
        seen_ranges: dict[tuple, ProteinEntry] = {}  # Track ranges to entries for deduplication
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header if present
            if has_header:
                next(csv_reader, None)
                
            for row_idx, row in enumerate(csv_reader):
                if len(row) <= column_index:
                    continue
                    
                # Reconstruct the original format: protein_id,ranges
                protein_id = row[column_index].strip()
                if not protein_id:
                    continue
                    
                # Get range part (if exists)
                range_part = ""
                if len(row) > column_index + 1:
                    range_part = row[column_index + 1].strip()
                
                # Reconstruct the full entry format
                if range_part:
                    raw_entry = f"{protein_id},{range_part}"
                else:
                    raw_entry = f"{protein_id},"
                    
                try:
                    entry = self._parse_complex_entry(raw_entry)
                    if entry:
                        # Check for range deduplication
                        if entry.ranges:
                            range_key = tuple(sorted(entry.ranges))
                            if range_key in seen_ranges:
                                # Skip duplicate range, keep first occurrence
                                continue
                            seen_ranges[range_key] = entry
                        
                        entries.append(entry)
                        
                except Exception as e:
                    # Log warning but continue processing
                    print(f"Warning: Error parsing row {row_idx + 1}: {raw_entry} - {e}")
                    continue
        
        return entries