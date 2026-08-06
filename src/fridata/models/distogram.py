from pathlib import Path
from typing import List

from pydantic import BaseModel

from fridata.models.distogramType import DistogramType


class Distogram(BaseModel):
    name: str
    thresholds: List[int] = [6]
    type: DistogramType = DistogramType.CA
    location: Path

    def generate(self):
        pass
