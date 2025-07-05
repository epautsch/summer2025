import json
from dataclasses import dataclass, field
from typing import List, Any
from enum import Enum, auto

from rich.console import Console

from agents.base_agent import BaseAgent
from agents.explainer_agent import ExplainerAgent
from core.executor import Executor
from core.action import Action, ActionType

console = Console()


class SessionState(Enum):
    INIT = auto()
    EXPLAINING = auto()
    QUIZZING = auto()
    CODING = auto()
    REVIEW = auto()
    FINISHED = auto()


@dataclass
class SessionAgent(BaseAgent):
    """
    Manages the overall session, coordinating between different agents.
    """ 
    executor:   Executor
    explainer:  ExplainerAgent

    lesson_topic: str = ""
    lesson_objectives: List[str] = field(default_factory=list)
    current_objective: str = ""
    default_topics: List[str] = field(default_factory=lambda: [
        "Introduction to Parallel Computing Concepts (shared memory vs. distributed memory)",
        "OpenMP: Parallelizing Loops with Directives",
        "CUDA: Vector Addition with Thrust",
        "MPI: Hello World and Basic Point-to-Point Communication",
        "SYCL: Simple Kernel for Array Multiplication"
    ])
    state:      SessionState = SessionState.INIT

    def step(self, user_input: str) -> Action:
        """
        1) Build a JSON-steering prompt for the SessionAgent
           describing current state + user_input.
        2) Call the LLM and parse its JSON -> Action.
        3) Update state if necessary.
        """
        prompt = self._build_state_prompt(user_input)
        raw = self.model.generate(prompt)
        #self.history.add(f"Session Prompt: {prompt}")
        #self.history.add(f"Session Response: {raw}")

        #data = json.loads(raw)
        #action = Action(type=ActionType[data["action"]], payload=data.get("payload", {}))
        action = self._parse_action(raw, expect=list(ActionType))

        self._transition_state(action)
        return action

    def handle(self, action: Action) -> Any:
        """
        Dispatch an Action to the appropriate sub-agent or executor.
        Returns the Observation or result of execution.
        """
        if action.type == ActionType.INITIALIZE:
            # Initialize the session with a lesson plan
            self.lesson_topic = action.payload.get("topic", "")
            self.lesson_objectives = action.payload.get("objectives", [])
            self.current_objective = self.lesson_objectives[0] if self.lesson_objectives else ""
            #sub = self.planner.create_plan_action(action.payload.get("topic", ""))
            obs = self.executor.execute(action)

        elif action.type == ActionType. NEXT_OBJECTIVE:
            sub = self.planner.next_objective_action()
            obs = self.executor.execute(sub)

        elif action.type == ActionType.EXPLAIN_CONCEPT:
            concept = action.payload.get("concept", "")
            is_question = action.payload.get("is_question", False)
            question = action.payload.get("question", "")
            if is_question:
                sub = self.explainer.answer_question_action(concept, question)
            else:
                sub = self.explainer.explain_concept_action(concept)
            obs = self.executor.execute(sub)

        elif action.type == ActionType.FINISH:
            obs = self.executor.execute(action)

        else:
            # fallback: treat any other as question to the explainer
            text = action.payload.get("text", "")
            sub = self.explainer.explain_concept_action(text)
            obs = self.executor.execute(sub)

        self.history.add(f"Observation: {getattr(obs, 'result', obs)}")
        return obs

    def _build_state_prompt(self, user_input: str) -> str:
        """
        Build a prompt that includes the current session state and user input.
        """

        state_prompt = (
            f"CURRENT STATE: {self.state.name}\n"
            f"TOPIC: {self.lesson_topic}\n"
            f"OBJECTIVES: {self.lesson_objectives}\n"
            f"CURRENT OBJECTIVE: {self.current_objective}\n"
            f"HISTORY: {self.history.get_full()}\n"
            f"USER INPUT: {user_input}\n"
            "Choose the next action based on the information above."
        )

        return state_prompt

        #plan_summary = (
        #    f"TOPIC: {self.planner.lesson_topic}\n"
        #    f"OBJECTIVES: {len(self.planner.lesson_objectives)} total\n"
        #    f"CURRENT: {self.planner.lesson_objectives[self.planner.current_index]}"
        #)
        #valid = ", ".join([a.name for a in ActionType])
        #return (
            #f"CURRENT STATE: {self.state.name}\n"
            #f"TOPIC:\n"
            #f"USER INPUT: {user_input}\n"
            #f"You may choose from the following actions: {valid}\n"
            #"Reply *only* with a JSON object {\"action\": ..., \"payload\": ...}."
        #)

    def _transition_state(self, action: Action):
        """
        Update the agent's own state variables when certain actions occur.
        """
        if action.type == ActionType.CREATE_LESSON_PLAN:
            self.state = SessionState.PLANNING
        elif action.type == ActionType.NEXT_OBJECTIVE:
            self.state = SessionState.EXPLAINING
        elif action.type == ActionType.EXPLAIN_CONCEPT:
            self.state = SessionState.EXPLAINING
        elif action.type == ActionType.FINISH:
            self.state = SessionState.FINISHED
        else:
            # fallback: stay in current state
            pass

    def run(self):
        """
        Interactive loop: read user input, call `step`, then `handle`, until finished.
        """
        from rich.prompt import Prompt

        console.print("[bold green]👋 Welcome to the HPC Tutor![/]")
        console.print("Please choose a topic to start your session.")
        topics = self.default_topics
        for idx, topic in enumerate(topics, start=1):
            console.print(f"[bold blue]{idx}. {topic}[/]")
        choice = Prompt.ask("Enter a number or type a new topic:")
        try:
            topic = topics[int(choice) - 1]
        except Exception:
            topic = choice.strip()
        console.print(f"[bold blue]Selected topic: {topic}[/]")

        #action = self.planner.create_plan_action(topic)
        #obs = self.executor.execute(action)
        #self.history.add(f"Observation: {obs.result}")

        action = self.step(f"Initialize lesson plan for topic: {topic}")
        self.handle(action)

        while self.state != SessionState.FINISHED:
            user_input = Prompt.ask("Your input")
            action = self.step(user_input)
            self.handle(action)
        console.print("[bold green]Session complete![/]")
