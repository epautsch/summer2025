from dataclasses import dataclass, field
from typing import List

from rich.console import Console
from rich.panel import Panel

from core.action import Action, ActionType
from agents.base_agent import BaseAgent

console = Console()


@dataclass
class LessonPlannerAgent(BaseAgent):
    """
    Agent responsible for creating and managing the lesson plan.
    Maintains its own history (just planning steps) and emits Actions
    to drive the overall session.
    """
    lesson_topic: str = ""
    lesson_objectives: List[str] = field(default_factory=list)
    current_index: int = 0
    default_topics: List[str] = field(default_factory=lambda: [
        "Introduction to Parallel Computing Concepts (shared memory vs. distributed memory)",
        "OpenMP: Parallelizing Loops with Directives",
        "CUDA: Vector Addition with Thrust",
        "MPI: Hello World and Basic Point-to-Point Communication",
        "SYCL: Simple Kernel for Array Multiplication"
    ])

    def create_plan_action(self, topic: str = "") -> Action:
        """
        Request a new lesson plan for the given topic.
        Emits CREATE_LESSON_PLAN with payload { topic, objectives }.
        """
        prompt = (
            f"The user wants to learn about '{topic}'. "
            "Generate JSON with 'action': 'CREATE_LESSON_PLAN' and payload "
            "{ 'topic': <topic>, 'objectives': [ ... ] }"
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.CREATE_LESSON_PLAN])
        payload = action.payload
        # initialize internal state
        self.lesson_topic = payload.get('topic', '')
        self.lesson_objectives = payload.get('objectives', [])
        self.current_index = 0
        return action

    def next_objective_action(self) -> Action:
        """
        Advance to the next objective. Emits NEXT_OBJECTIVE with
        payload { new_index }.
        """
        new_idx = min(self.current_index + 1, len(self.lesson_objectives) - 1)
        prompt = (
            f"Proceed to the next objective index {new_idx}. "
            "Reply with JSON: { 'action': 'NEXT_OBJECTIVE', 'payload': { 'new_index': <new_idx> } }."
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.NEXT_OBJECTIVE])
        # update internam pointer
        idx = action.payload.get('new_index', self.current_index)
        if 0 <= idx < len(self.lesson_objectives):
            self.current_index = idx
        return action

    def call_explainer_action(self, text: str, is_question: bool = False) -> Action:
        """
        Delegate an explanation or follow-up question to the ExplainerAgent.
        Emits EXPLAIN_CONCEPT with payload { concept: ..., explanation: ..., examples: ? }.
        """
        if is_question:
            prompt = (
                f"Learner asks about '{text}' on objective '{self.lesson_objectives[self.current_index]}'. "
                "Repond with JSON: { 'action': 'EXPLAIN_CONCEPT', 'payload': "
                "{ 'explanation': <str>, 'examples': [ ... ] } }."
            )
        else:
            prompt = (
                f"Explain the concept '{text}' clearly, optionally with examples. "
                "Return JSON: { 'action': 'EXPLAIN_CONCEPT', 'payload': "
                "{ 'explanation': <str>, 'examples': [ ... ] } }."
            )

        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.EXPLAIN_CONCEPT])
        return action

    def show_plan(self):
        """
        Render the current lesson plan and highlight the current objective.
        """
        title = f"Lesson Plan: {self.lesson_topic}"
        lines = []
        for idx, obj in enumerate(self.lesson_objectives):
            prefix = "→ " if idx == self.current_index else " "
            lines.append(f"{prefix}{idx + 1}. {obj}")
        console.print(Panel("\n".join(lines), title=title))
