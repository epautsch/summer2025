from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from actions import Action, ActionType
from agents import LessonPlannerAgent, ExplainerAgent
from executor import Executor
from history import HistoryManager

console = Console()


@dataclass
class SessionManager:
    planner: LessonPlannerAgent
    explainer: ExplainerAgent
    executor: Executor
    history: HistoryManager

    def run(self):
        console.print("[bold green]👋 Welcome to the HPC Tutor![/]")
        # topic selection
        topics = self.planner.default_topics
        for idx, topic in enumerate(self.planner.default_topics, start=1):
            console.print(f"{idx}. {topic}")
        choice = Prompt.ask("Choose a topic by number or type a new one")
        try:
            topic = topics[int(choice) - 1]
        except Exception:
            topic = choice
        console.print(f"[red][DEBUG TOPIC CHOICE] You chose {topic}[/].")
        console.print("[red][DEBUG TOPIC CHOICE END][/]\n")

        # create and set lesson plan
        lesson_plan_prompt = self.planner.create_lesson_plan(topic)
        lesson_plan_json = self._call_llm(lesson_plan_prompt)
        # TODO need better error parsing here for failed conditional
        if lesson_plan_json["action"] == "CREATE_LESSON_PLAN":
            real_topic = lesson_plan_json["payload"]["topic"]
            objectives = lesson_plan_json["payload"]["objectives"]
            self.planner.set_plan(topic, objectives)
            console.print(f"[green] ✅ Saved lesson plan for \"{real_topic}\" with {len(objectives)} objectives.[/]\n")
            action = Action(
                type=ActionType.CREATE_LESSON_PLAN,
                payload={"topic": real_topic, "objectives": objectives}
            )
            obs = self.executor.execute(action)

            self.history.add(f"Observation: {obs.result}")
        else:
            console.print("[red] ✖ Unexpected response—couldn't create lesson plan.[/]")

        # loop through objectives
        for obj in objectives:
            console.print(Rule(f"📖 {obj}"))
            
            # generate explanation
            explanation_prompt = self.planner.explain_concept(obj)
            explanation_raw = self._call_llm(explanation_prompt)
            # parse json for explanation
            explanation, examples = self._extract_explanation_data(explanation_raw)

            action = Action(
                type=ActionType.EXPLAIN_CONCEPT,
                payload={
                    "concept": obj,
                    "explanation": explanation,
                    "examples": examples,
                }
            )
            obs = self.executor.execute(action)
            self.history.add(f"Observation: {obs.result}")

            # ask user for followup explanations before continuing
            while True:
                user_q = Prompt.ask(
                    "\nHave any questions? Type your question, or 'next' to continue"
                ).strip()
                if user_q.lower() in ("next", "n"):
                    break

                answer_prompt = self.planner.answer_question(obj, user_q)
                answer_raw = self._call_llm(answer_prompt)
                answer, examples = self._extract_explanation_data(answer_raw)

                action = Action(
                    type=ActionType.EXPLAIN_CONCEPT,
                    payload={
                        "concept": obj,
                        "explanation": answer,
                        "examples": examples,
                    }
                )
                obs = self.executor.execute(action)
                self.history.add(f"Observation: {obs.result}")
            
        console.print(Rule("🏁 Lesson Complete!"))
        summary = self.planner.llm.generate(f"Summarize the lesson on {topic} using {backend} and key takeaways.")
        console.print(Panel(summary, title="Lesson Summary"))
        hw = self.planner.llm.generate(f"Generate 4 homework exercises for {topic} on {backend}.")
        console.print(Panel(hw, title="Homework"))


