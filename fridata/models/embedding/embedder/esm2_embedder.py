import torch  
from transformers import AutoTokenizer, AutoModelForMaskedLM  
from .base_embedder import BaseEmbedder

# Parameters  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32  

class ESM2Embedder(BaseEmbedder):
    def __init__(self, device=None, batch_size=1000, model_name='esm2_t33_650M_UR50D'):
        super().__init__(device, batch_size)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_name}").to(self.device).to(torch.float32)

    def get_embedding(self, prot_id, prot_seq):
        inputs = self.tokenizer(prot_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        return embeddings[0,:].to('cpu').detach().to(torch.float32).numpy()
