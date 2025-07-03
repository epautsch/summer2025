from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any


class ActionType(Enum):
    CREATE_LESSON_PLAN = auto()
    NEXT_OBJECTIVE = auto()
    PREVIOUS_OBJECTIVE = auto()
    EXPLAIN_CONCEPT = auto()
    QUIZ_USER = auto()
    CODE = auto()
    SYSTEM_CALL = auto()
    GENERATE_HOMEWORK = auto()
    FINISH = auto()


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]
