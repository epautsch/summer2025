from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any


class ActionType(Enum):
    # Used by SessionAgent
    INITIALIZE = auto()
    CALL_EXPLAINER = auto()
    CALL_QUIZZER = auto()
    CALL_CODER = auto()
    CALL_REVIEWER = auto()
    QUERY_USER = auto()
    GENERATE_HOMEWORK = auto()
    FINISH = auto()

    # Used by ExplainerAgent
    EXPLAIN_CONCEPT = auto()

    # Used by QuizzerAgent
    GENERATE_QUIZ = auto()
    EVALUATE_QUIZ_ANSWER = auto()

    # Used by CoderAgent
    GENERATE_CODE = auto()

    # Used by BuilderAgent
    COMPILE_CODE = auto()
    RUN_CODE = auto()

    # Used by ReviewerAgent
    SYSTEM_CALL = auto()
    REVIEW_FINISH = auto()


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]
