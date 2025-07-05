from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any


class ActionType(Enum):
    INITIALIZE = auto()
    CALL_EXPLAINER = auto()
    CALL_QUIZER = auto()
    CALL_CODER = auto()
    SYSTEM_CALL = auto()
    GENERATE_HOMEWORK = auto()
    FINISH = auto()


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]
