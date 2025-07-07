from dataclasses import dataclass

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class QuizzerAgent(BaseAgent):
    """
    Agent that specializes in generating quizzes based on the current lesson topic.
    Maintains its own history and emits QUIZ actions.
    """

    def generate_quiz_action(self, topic: str) -> Action:
        """
        Generate a quiz for the given topic.
        Emits QUIZ action with payload { question: List[str], answers: List[str] }.
        """
        prompt = (
            f"Create a quiz question for the topic '{topic}'. "
            "Return JSON: { 'action': 'QUIZ', 'payload': "
            "{ 'question': [ ... ], 'answers': [ ... ] } }."
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.GENERATE_QUIZ])
        return action
