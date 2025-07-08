import os
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.prompt import Prompt

from core.action import Action, ActionType
from core.utilities import save_to_file, run_shell
from core.observation import Observation

console = Console()


@dataclass
class Executor:
    def execute(self, action: Action) -> Observation:
        console.log(f"[bold cyan]Executing action[/] → {action.type.name}")
        p = action.payload

        if action.type == ActionType.INITIALIZE:
            # payload: {"topic": ..., "objectives": [...[}
            title = f"Lesson Plan: {p['topic']}"
            body = "\n".join(f"- {o}" for o in p["objectives"])
            console.print(Panel(body, title=title))
            console.print("Would you like to make any changes to the plan, or are you ready to start?")
            return Observation(result=f"Lesson plan created for {p['topic']}.")

        elif action.type == ActionType.EXPLAIN_CONCEPT:
            # payload: {"concept": ..., "explanation": "..."}
            title = f"Concept: {p['concept']}"
            console.print(Panel(p["explanation"], title=title))

            examples = p.get("examples", [])
            if examples:
                table = Table(title="Examples")
                table.add_column("Examples", style="italic")
                for ex in examples:
                    table.add_row(ex)
                console.print(table)
            return Observation(result="Displayed explanation + examples.")

        elif action.type == ActionType.GENERATE_QUIZ:
            # payload: {"concept": ..., "question": "...", "options": [...], "correct_option_index": ...}
            title = f"Quiz Question on {p['concept']}"
            question = p["question"]
            options = p["options"]

            table = Table(title=title)
            table.add_column("Question", style="bold")
            table.add_column("Options")
            table.add_row(question, "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options)))
            console.print(table)

            return Observation(result=f"Quiz generated for {p['concept']}.")

        elif action.type == ActionType.EVALUATE_QUIZ_ANSWER:
            p = action.payload
            correct = p.get("is_correct", False)
            feedback = p.get("feedback", "")

            status = "[bold green]✔ Correct![/]" if correct else "[bold red]✘ Incorrect.[/]"
            console.print(Panel(status, title="Quiz Result", expand=False))
            console.print(Panel(feedback, title="Feedback", expand=False))

            console.print("Are you ready to continue with the lesson?")

            return Observation(result=f"""
                               Quiz answer evaluated. Answer was {'correct' if correct else 'incorrect'}.
                               User prompted if they are ready to continue.
                               """)

        elif action.type == ActionType.CODE:
            code = action.payload.get('input', '')
            fname = action.payload.get('filename', 'code.out')
            ext = os.path.splitext(fname)[1].lstrip('.')
            lang = ext if ext else 'text'
            syntax = Syntax(code, lang, line_numbers=True)
            console.print(Panel(syntax, title=f"Generated Code → {fname}"))
            save_to_file(code, fname)
            return Observation(result=f"Saved code to {fname}")

        elif action.type == ActionType.SYSTEM_CALL:
            if isinstance(action.payload, str):
                cmd = action.payload
            else:
                cmd = action.payload.get('cmd', '')
            output = run_shell(cmd)
            return Observation(result=output)
