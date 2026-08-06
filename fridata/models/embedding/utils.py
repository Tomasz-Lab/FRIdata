from pathlib import Path
from typing import Dict
import torch
import h5py

def save_batch(output_path: Path, batch_index: int, embeddings_pure: Dict[str, torch.Tensor]):  
    """Function to save a batch of embeddings to an H5 file."""  
    batch_file = output_path / f"batch_{batch_index}.h5"  
    with h5py.File(batch_file, 'w') as f:  
        for seq_id, embedding in embeddings_pure.items():  
            f.create_dataset(seq_id, data=embedding)  