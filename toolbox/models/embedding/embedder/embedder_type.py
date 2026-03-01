from enum import Enum
from toolbox.models.embedding.embedder.esm2_embedder import ESM2Embedder
from toolbox.models.embedding.embedder.esmc_embedder import ESMCEmbedder
from toolbox.models.embedding.embedder.base_embedder import BaseEmbedder

class EmbedderType(Enum):
    ESM2_T30_150M = ("esm2_t30_150M_UR50D", ESM2Embedder, 640)
    ESM2_T33_650M = ("esm2_t33_650M_UR50D", ESM2Embedder, 1280)
    ESMC_300M = ("esmc_300m", ESMCEmbedder, 960)
    ESMC_600M = ("esmc_600m", ESMCEmbedder, 1152)

    def __init__(self, value, embedder_class: BaseEmbedder, embedding_size: int):
        self._value_ = value
        self.embedder_class: BaseEmbedder = embedder_class
        self.embedding_size: int = embedding_size