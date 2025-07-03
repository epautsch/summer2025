import json
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console

from core.model import LLMClient
from core.history_manager import HistoryManager
from core.action import Action, ActionType
from core.utilities import strip_markdown_fences

console = Console()


@dataclass
class BaseAgent:
    """
    Base class for all agents, encapsulating common LLM interaction
    and history management.
    """
    model: LLMClient
    history: HistoryManager

    def _generate(self, prompt: str) -> str:
        """
        Prepend this agent's history to the prompt, invoke the LLM,
        and log the interaction. Returns the raw JSON string from the model.
        """
        context = self.history.get_full()
        full_prompt = f"{context}\n{prompt}" if context else prompt
        raw = self.model.generate(full_prompt)
        self.history.add(f"Prompt: {prompt}")
        self.history.add(f"Response: {raw}")
        return raw

    def _parse_action(self, raw: str,
                      expect: Optional[List[ActionType]] = None) -> Action:
        """
        Strip markdown fences, parse JSON into an Action, validating its type.
        """
        text = strip_markdown_fences(raw)
        data = json.loads(text)
        act_type = ActionType[data['action']]
        if expect and act_type not in expect:
            raise ValueError(f"Unexpected action type: {act_type}, expected one of {expect}")
        action = Action(type=act_type, payload=data.get('payload', {}))
        return action
