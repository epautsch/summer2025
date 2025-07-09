from dataclasses import dataclass

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class ReviewerAgent(BaseAgent):
    """
    Agent that specializes in compiling, running, and reviewing code files.
    """

    def initialize_review_action(self, file_name: str, topic: str) -> Action:
        """
        Initialize a review for the given file and topic.
        Emits SYSTEM_CALL action with payload { file_name: str, topic: str }.
        """
        prompt = (
            f"""
            Review the code in '{file_name}' related to the topic '{topic}'.
            """
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.SYSTEM_CALL])

        return action

    def step(self) -> Action:
        """
        Perform a single step in the review process.
        This method should be called repeatedly until a REVIEW_FINISH action is returned.
        """
        prompt = "Please provide the next step in the code review process."
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.SYSTEM_CALL, ActionType.REVIEW_FINISH])

        return action
