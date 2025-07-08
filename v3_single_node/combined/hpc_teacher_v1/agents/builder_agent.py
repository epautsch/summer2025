from dataclass import dataclass

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class BuilderAgent(BaseAgent):
    """
    Agent that specializes in compiling and running code files.
    """

    def compile_code(self, file_name: str) -> Action:
        """
        Compile the code in the specified file.
        """
        prompt = (
            f"""
            Compile the code in the following file:
            {file_name}
            """
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.COMPILE_CODE])

        return action
