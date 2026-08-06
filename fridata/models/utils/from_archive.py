from pathlib import Path
from typing import List, Tuple


from fridata.models.manage_dataset.utils import cif_to_pdbs, compress_and_save_h5

from fridata.utlis.logging import logger


def extract_batch_from_archive(batch_files: List[Path], batch_dir: Path) -> Tuple[List[str], str]:
        """Process a batch of CIF files and save to HDF5"""
        all_pdbs = []
        all_contents = []
        
        for file in batch_files:
            try:
                # Read CIF content
                cif_content = file.read_text()
                # Convert CIF to PDB
                converted_pdbs, _ = cif_to_pdbs((cif_content, file.stem))
                
                # Add converted PDBs to lists
                for pdb_id, pdb_content in converted_pdbs.items():
                    all_pdbs.append(pdb_id)
                    all_contents.append(pdb_content)
                    
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                continue
        
        if all_pdbs:
            # Save batch to HDF5
            h5_path = compress_and_save_h5(batch_dir, (all_pdbs, all_contents, []))
            logger.info(f"Extracted {len(all_pdbs)} structures")
            return all_pdbs, h5_path
        return [], ""