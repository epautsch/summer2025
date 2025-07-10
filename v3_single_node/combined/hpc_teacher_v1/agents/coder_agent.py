from dataclasses import dataclass

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class CoderAgent(BaseAgent):
    """
    Agent that specializes in generating code based on the current lesson
    direction given by the SessionAgent.
    """

    def generate_code_action(self, code_direction: str, file_name: str) -> Action:
        """
        Generate code based on the provided direction from the SessionAgent.
        Emits GENERATE_CODE action with payload { code: str, file_name: str }.
        """
        prompt = (
            f"""
            Generate code based on the following direction:
            {code_direction}
            Use the file name '{file_name}' for the generated code.
            """
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.GENERATE_CODE])

        return action
