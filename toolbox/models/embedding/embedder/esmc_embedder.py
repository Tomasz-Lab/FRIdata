from typing import Dict
from pathlib import Path
import torch
import gc
from multiprocessing import Process
from toolbox.models.embedding.utils import save_batch
from toolbox.utlis.logging import logger
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from .base_embedder import BaseEmbedder

batch_size = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ESMCEmbedder(BaseEmbedder):
    def __init__(self, device=None, batch_size=1000, model_name="esmc_600m"):
        super().__init__(device, batch_size)
        self.model = ESMC.from_pretrained(model_name).to(self.device)

    def get_embedding(self, prot_id, prot_seq):
        protein = ESMProtein(sequence=prot_seq)
        protein_tensor = self.model.encode(protein)
        logits_output = self.model.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        return logits_output.embeddings[0,:,:].to('cpu').detach().to(torch.float32).numpy()

def embed(
    sequences: Dict[str, str],  
    output_path: Path,  
    embedding_batch_size: int=batch_size,
):
    save_batch_processes = []
    model = ESMC.from_pretrained("esmc_600m").to(device)
    logger.info(f"Embeddings are computed using {str(device).upper()}")
    try:
        model.eval()
        with torch.inference_mode():
            embeddings_pure_batch = {}
            batch_index = 0
            for prot_id, prot_seq in sequences.items():
                protein = ESMProtein(sequence=prot_seq)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings_pure = logits_output.embeddings[0,:,:].to('cpu').detach().to(torch.float32).numpy()
                assert len(prot_seq) == embeddings_pure.shape[0], f'Invalid character in {prot_id}'
                embeddings_pure_batch[prot_id] = embeddings_pure
                # Clear GPU memory after each iteration
                del logits_output, protein_tensor, protein
                torch.cuda.empty_cache()
                if len(embeddings_pure_batch) >= embedding_batch_size:
                    p = Process(target=save_batch, args=(output_path, batch_index, embeddings_pure_batch.copy()))
                    p.start()
                    save_batch_processes.append((p, output_path / f"batch_{batch_index}.h5", list(embeddings_pure_batch.keys())))
                    embeddings_pure_batch.clear()
                    batch_index += 1
            # Save any remaining embeddings
            if embeddings_pure_batch:
                p = Process(target=save_batch, args=(output_path, batch_index, embeddings_pure_batch.copy()))
                p.start()
                save_batch_processes.append((p, output_path / f"batch_{batch_index}.h5", list(embeddings_pure_batch.keys())))
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
    final_index = {}
    for p, file_path, ids in save_batch_processes:
        p.join()
        for prot_id in ids:
            final_index[prot_id] = str(file_path)
    return final_index


