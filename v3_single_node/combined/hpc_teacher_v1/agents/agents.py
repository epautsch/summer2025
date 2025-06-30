import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from core.model import ManagerModel
from core.history_manager import HistoryManager
from core.action import Action, ActionType
from core.utilities import strip_markdown_fences

console.Console()


@dataclass
class BaseAgent:
    """
    Base class for all agents, encapsulating common LLM interaction and history management.
    """
    model: ManagerModel
    history: HistoryManager

    def _generate(self, prompt: str) -> str:
        """
        Prepend this agent's history to the prompt, invoke the LLM, and log the interaction.
        Returns the raw JSON string from the model.
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


@dataclass
class LessonPlannerAgent(BaseAgent):
    """
    Agent responsible for creating and managing the lesson plan.
    Maintains its own history (just planning steps) and emits Actions
    to drive the overall session.
    """
    model: ManagerModel
    history: HistoryManager
    lesson_topic: str = ""
    lesson_objectives: List[str] = field(default_factory=list)
    default_topics: List[str] = field(default_factory=lambda: [
        "Introduction to Parallel Computing Concepts (shared memory vs. distributed memory)",
        "OpenMP: Parallelizing Loops with Directives",
        "CUDA: Vector Addition with Thrust",
        "MPI: Hello World and Basic Point-to-Point Communication",
        "SYCL: Simple Kernel for Array Multiplication"
    ])

    def set_plan(self, topic: str, objectives: List[str]) -> None:
        self.lesson_topic = topic
        self.lesson_objectives = objectives

     # MARKED for removal
    """
    def suggest_topics(self) -> List[str]:
        prompt = (
            "List 5 beginner-friendly HPC topics across different backends (CUDA, OpenMP, "
            "MPI, SYCL, etc.), each on its own line."
        )
        resp = self.llm.generate(prompt)
        topics = [t.strip() for t in resp.splitlines() if t.strip()]
        return topics or self.default_topics
    """

    def create_lesson_plan(self, topic: str) -> List[str]:
        return (
            f"The user has chosen to learn about {topic}. "
            "Create the lesson plan as JSON with keys 'thought', 'action', and 'payload'."
        )

    def explain_concept(self, concept: str) -> str:
        return f"Explain the concept: {concept}"

    def answer_question(self, concept: str, question: str) -> str:
        return (
            f"The learner is asking a follow-up about '{concept}':\n"
            f"\"{question}\"\n"
            "Please answer clearly and concisely."
        )

