from enum import Enum


class PieceType(Enum):
    UNKNOWN=-1
    LEFT_UP=0
    RIGHT_UP=1
    LEFT_DOWN=2
    RIGHT_DOWN=3
    CENTER_UP=4
    CENTER_DOWN=5
    CENTER_LEFT=6
    CENTER_RIGHT=7
    CENTER=8
