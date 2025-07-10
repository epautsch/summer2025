from dataclasses import dataclass, field
from typing import List, Any
from enum import Enum, auto

from rich.console import Console

from agents.base_agent import BaseAgent
from agents.explainer_agent import ExplainerAgent
from agents.quizzer_agent import QuizzerAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
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
    quizzer:    QuizzerAgent
    coder:      CoderAgent
    reviewer:   ReviewerAgent

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
    state:      SessionState = SessionState.INIT

    def step(self, user_input: str) -> Action:
        """
        1) Build a JSON-steering prompt for the SessionAgent
           describing current state + user_input.
        2) Call the LLM and parse its JSON -> Action.
        3) Update state if necessary.
        """
        prompt = self._build_state_prompt(user_input)
        raw = self._generate(prompt)

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
            obs = self.executor.execute(action)

        elif action.type == ActionType.CALL_EXPLAINER:
            concept = action.payload.get("concept", "")
            is_question = action.payload.get("is_question", False)
            question = action.payload.get("question", "")

            try:
                idx = self.lesson_objectives.index(concept)
            except ValueError:
                console.print(f"[bold red]Error: Concept '{concept}' not found in lesson objectives.[/]")
                return None
            else:
                self.current_index = idx

            if is_question:
                sub = self.explainer.answer_question_action(concept, question)
            else:
                sub = self.explainer.explain_concept_action(concept)
            obs = self.executor.execute(sub)

        elif action.type == ActionType.CALL_QUIZZER:
            payload = action.payload

            if "concept" in payload:
                self.current_index = self.lesson_objectives.index(payload["concept"])
                sub = self.quizzer.generate_quiz_action(payload["concept"])
            elif "user_answer" in payload:
                sub = self.quizzer.evaluate_quiz_answer_action(payload["user_answer"])
            else:
                raise ValueError("Quiz action must have either 'concept' or 'user_answer' in payload.")

            obs = self.executor.execute(sub)

        elif action.type == ActionType.CALL_CODER:
            code_direction = action.payload.get("code_direction", "")
            file_name = action.payload.get("file_name", "")

            sub = self.coder.generate_code_action(code_direction, file_name)
            obs = self.executor.execute(sub)

        elif action.type == ActionType.CALL_REVIEWER:
            payload = action.payload
            file_name = payload.get("file_name", "")
            topic = payload.get("topic", self.lesson_topic)

            review_action = self.reviewer.initialize_review_action(
                file_name=file_name,
                topic=topic
            )

            while review_action.type != ActionType.REVIEW_FINISH:
                # print type of review_action
                console.print(f"[bold cyan]Review Action Type:[/bold cyan] {review_action.type}")

                obs = self.executor.execute(review_action)

                self.reviewer.history.add(f"Reviewer Observation: {obs.result}")

                review_action = self.reviewer.step()

            obs = self.executor.execute(review_action)

        elif action.type == ActionType.QUERY_USER:
            # This action is used to query the user for input
            obs = self.executor.execute(action)

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
            f"CURRENT OBJECTIVE: {self.lesson_objectives[self.current_index] if self.lesson_objectives else None}\n"
            f"HISTORY: {self.history.get_full()}\n"
            f"USER INPUT: {user_input}\n"
            #"Choose the next action based on the information above."
        )

        return state_prompt

    def _transition_state(self, action: Action):
        """
        Update the agent's own state variables when certain actions occur.
        """
        if action.type == ActionType.INITIALIZE:
            self.state = SessionState.INIT
        elif action.type == ActionType.CALL_EXPLAINER:
            self.state = SessionState.EXPLAINING
        elif action.type == ActionType.CALL_QUIZZER:
            self.state = SessionState.QUIZZING
        elif action.type == ActionType.CALL_CODER:
            self.state = SessionState.CODING
        elif action.type == ActionType.CALL_REVIEWER:
            self.state = SessionState.REVIEW
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
        choice = Prompt.ask("Enter a number or type a new topic")
        try:
            topic = topics[int(choice) - 1]
        except Exception:
            topic = choice.strip()
        console.print(f"[bold blue]Selected topic: {topic}[/]")

        action = self.step(f"Initialize lesson plan for topic: {topic}")
        self.handle(action)

        while self.state != SessionState.FINISHED:
            user_input = ""
            while not user_input.strip():
                user_input = Prompt.ask("Your input").strip()
            action = self.step(user_input)
            self.handle(action)

        console.print("[bold green]Session complete![/]")
