from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from fridata.models.embedding.embedder.base_embedder import BaseEmbedder


_EMBEDDINGS_EXTRA_HINT = (
    "Embedding generation requires the optional 'embeddings' dependencies "
    "(torch, esm, transformers). Install them with:\n"
    "    pip install 'FRIdata[embeddings]'\n"
    "or, for a specific CUDA version / CPU-only build, run scripts/install_pytorch.sh."
)


def _lazy_embedder_class(member: "EmbedderType") -> Type[BaseEmbedder]:
    name = member.name
    try:
        if name.startswith("ESM2"):
            from fridata.models.embedding.embedder.esm2_embedder import ESM2Embedder

            return ESM2Embedder
        if name.startswith("ESMC"):
            from fridata.models.embedding.embedder.esmc_embedder import ESMCEmbedder

            return ESMCEmbedder
        if name.startswith("GLM2"):
            from fridata.models.embedding.embedder.glm2_embedder import GLM2Embedder

            return GLM2Embedder
    except ImportError as exc:
        raise ImportError(f"{exc}\n\n{_EMBEDDINGS_EXTRA_HINT}") from exc
    raise ValueError(f"Unknown embedder kind for {name!r}")


class EmbedderType(Enum):
    ESM2_T30_150M = ("esm2_t30_150M_UR50D", 640)
    ESM2_T33_650M = ("esm2_t33_650M_UR50D", 1280)
    ESMC_300M = ("esmc_300m", 960)
    ESMC_600M = ("esmc_600m", 1152)
    GLM2_150M = ("gLM2_150M", 640)
    GLM2_650M = ("gLM2_650M", 1280)

    def __init__(self, model_id: str, embedding_size: int):
        self._value_ = model_id
        self.embedding_size = embedding_size

    @property
    def embedder_class(self) -> Type[BaseEmbedder]:
        return _lazy_embedder_class(self)

    def create_embedder(self) -> BaseEmbedder:
        return self.embedder_class(model_name=self.value)
