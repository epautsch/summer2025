from dataclasses import dataclass, field
from typing import Dict, Any

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class QuizzerAgent(BaseAgent):
    """
    Agent that specializes in generating quizzes based on the current lesson topic.
    Maintains its own history and emits QUIZ actions.
    """
    last_quiz: Dict[str, Any] = field(default_factory=dict)

    def generate_quiz_action(self, topic: str) -> Action:
        """
        Generate a quiz for the given topic.
        Emits QUIZ action with payload { question: List[str], answers: List[str] }.
        """
        prompt = (
            f"""
            Create a quiz question for the topic '{topic}'.
            """
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.GENERATE_QUIZ])

        self.last_quiz = action.payload.copy()

        return action

    def evaluate_quiz_answer_action(self, user_answer: str) -> Action:
        """
        Evaluate the user's answer against the last generated quiz.
        """
        if not self.last_quiz:
            raise RuntimeError("No quiz has been generated yet. Call generate_quiz_action first.")

        q = self.last_quiz["question"]
        options = self.last_quiz["options"]
        correct_index = self.last_quiz["correct_option_index"]

        prompt = (
            f"""
            Evaluate the user's answer.

            Question: {q}

            Options:
            1) {options[0]}
            2) {options[1]}
            3) {options[2]}
            4) {options[3]}

            The correct option is {correct_index}.

            User's answer: {user_answer}

            Please respond with a JSON object.
            """
        )

        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.EVALUATE_QUIZ_ANSWER])
        return action
