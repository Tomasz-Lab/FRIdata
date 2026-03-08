import torch
from transformers import AutoModel, AutoTokenizer
from .base_embedder import BaseEmbedder

PREP_SIGN = "<+>"


class GLM2Embedder(BaseEmbedder):
    def __init__(self, device=None, batch_size=1000, model_name="gLM2_150M"):
        super().__init__(device, batch_size)
        self.model_name = model_name
        use_bfloat16 = self.device.type == "cuda"
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"tattabio/{model_name}", trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                f"tattabio/{model_name}",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(self.device)
        )

    def get_embedding(self, prot_id, prot_seq):
        sequence = PREP_SIGN + prot_seq
        inputs = self.tokenizer([sequence], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs["input_ids"], output_hidden_states=True)
        embeddings = outputs.last_hidden_state[0]
        return embeddings.to("cpu").detach().to(torch.float32).numpy()

    def validate_embedding(self, prot_seq, embeddings):
        return len(prot_seq) + 1 == embeddings.shape[0]
