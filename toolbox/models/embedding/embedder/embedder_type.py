from enum import Enum
from toolbox.models.embedding.embedder.esm2_embedder import ESM2Embedder
from toolbox.models.embedding.embedder.esmc_embedder import ESMCEmbedder
from toolbox.models.embedding.embedder.glm2_embedder import GLM2Embedder
from toolbox.models.embedding.embedder.base_embedder import BaseEmbedder

class EmbedderType(Enum):
    ESM2_T30_150M = ("esm2_t30_150M_UR50D", ESM2Embedder, 640)
    ESM2_T33_650M = ("esm2_t33_650M_UR50D", ESM2Embedder, 1280)
    ESMC_300M = ("esmc_300m", ESMCEmbedder, 960)
    ESMC_600M = ("esmc_600m", ESMCEmbedder, 1152)
    GLM2_150M = ("gLM2_150M", GLM2Embedder, 640)
    GLM2_650M = ("gLM2_650M", GLM2Embedder, 1280)

    def __init__(self, value, embedder_class: type[BaseEmbedder], embedding_size: int):
        self._value_ = value
        self.embedder_class: type[BaseEmbedder] = embedder_class
        self.embedding_size: int = embedding_size

    def create_embedder(self) -> BaseEmbedder:
        return self.embedder_class(model_name=self.value)