from abc import ABC, abstractmethod
from multiprocessing import Process
import torch
import gc
from toolbox.models.embedding.utils import save_batch
from toolbox.utlis.logging import logger

class BaseEmbedder(ABC):
    def __init__(self, device=None, batch_size=1000):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

    @abstractmethod
    def get_embedding(self, prot_id, prot_seq):
        pass

    def embed(self, sequences, output_path):
        save_batch_processes = []
        logger.info(f"Embeddings are computed using {str(self.device).upper()}")
        try:
            self.model.eval()
            with torch.inference_mode():
                embeddings_pure_batch = {}
                batch_index = 0
                for prot_id, prot_seq in sequences.items():
                    embeddings_pure = self.get_embedding(prot_id, prot_seq)
                    assert self.validate_embedding(prot_seq, embeddings_pure), f'Invalid embedding shape for {prot_id}: expected shape[0] to match sequence length'
                    embeddings_pure_batch[prot_id] = embeddings_pure
                    torch.cuda.empty_cache()
                    if len(embeddings_pure_batch) >= self.batch_size:
                        p = Process(target=save_batch, args=(output_path, batch_index, embeddings_pure_batch.copy()))
                        p.start()
                        save_batch_processes.append((p, output_path / f"batch_{batch_index}.h5", list(embeddings_pure_batch.keys())))
                        embeddings_pure_batch.clear()
                        batch_index += 1
                if embeddings_pure_batch:
                    p = Process(target=save_batch, args=(output_path, batch_index, embeddings_pure_batch.copy()))
                    p.start()
                    save_batch_processes.append((p, output_path / f"batch_{batch_index}.h5", list(embeddings_pure_batch.keys())))
        finally:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        final_index = {}
        for p, file_path, ids in save_batch_processes:
            p.join()
            for prot_id in ids:
                final_index[prot_id] = str(file_path)
        return final_index

    def validate_embedding(self, prot_seq, embeddings):
        return len(prot_seq) + 2 == embeddings.shape[0]