from pydantic import BaseModel


class PressureInput(BaseModel):
    """Runtime features collected from one Tetris-like game state"""

    # Field names intentionally match the JSON payload and exported feature schema
    boardHeightRatio: float
    holesRatio: float
    bumpiness: float
    deepestWell: float
    piecesSinceLastClear: float
    hardDropRate: float
    softDropRate: float
    rotationRate: float
    averageTimeToPlace: float
    currentNoise: float
    currentComposure: float
    roundProgress: float
