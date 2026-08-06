from enum import Enum


class DatabaseType(Enum):
    PDB = "PDB"
    AFDB = "AFDB"
    ESMatlas = "ESMatlas"
    other = "other"
