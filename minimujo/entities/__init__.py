from enum import Enum
from typing import Union
from minigrid.core.world_object import WorldObj
from minigrid.core.constants import COLOR_TO_IDX

class ObjectEnum(Enum):
    BALL = 0
    BOX = 1
    DOOR = 2
    KEY = 3
    
    @classmethod
    def get_id(cls, obj: Union[str, WorldObj]):
        if isinstance(obj, WorldObj):
            obj = str(obj)
        try:
            return cls[obj.upper()].value
        except KeyError:
            raise ValueError(f"{obj} is not a valid object name")
        
OBJECT_NAMES = ["ball", "box", "key", "door"]
        
def get_color_id(color: str):
    try:
        return COLOR_TO_IDX.get(color)
    except ValueError:
        raise ValueError(f"{color} is not an entity color")