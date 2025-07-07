from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any


class ActionType(Enum):
    # Used by SessionAgent
    INITIALIZE = auto()
    CALL_EXPLAINER = auto()
    CALL_QUIZZER = auto()
    CALL_CODER = auto()
    SYSTEM_CALL = auto()
    GENERATE_HOMEWORK = auto()
    FINISH = auto()

    # Used by ExplainerAgent
    EXPLAIN_CONCEPT = auto()

    # Used by QuizzerAgent
    GENERATE_QUIZ = auto()
    EVALUATE_QUIZ_ANSWER = auto()


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]
