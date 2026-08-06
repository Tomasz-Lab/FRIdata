import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from .base_embedder import BaseEmbedder


class ESMCEmbedder(BaseEmbedder):
    def __init__(self, device=None, batch_size=1000, model_name="esmc_600m"):
        super().__init__(device, batch_size)
        self.model = ESMC.from_pretrained(model_name).to(self.device)

    def get_embedding(self, prot_id, prot_seq):
        protein = ESMProtein(sequence=prot_seq)
        self.model.eval()
        with torch.inference_mode():
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
        return logits_output.embeddings[0,:,:].to('cpu').detach().to(torch.float32).numpy()
