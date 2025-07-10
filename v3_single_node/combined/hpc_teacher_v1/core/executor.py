import os
import subprocess
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
            correct = bool(p.get("is_correct", "False"))
            feedback = p.get("feedback", "")

            status = "[bold green]✔ Correct![/]" if correct else "[bold red]✘ Incorrect.[/]"
            console.print(Panel(status, title="Quiz Result", expand=False))
            console.print(Panel(feedback, title="Feedback", expand=False))

            console.print("Are you ready to continue with the lesson?")

            return Observation(result=f"""
                               Quiz answer evaluated. Answer was {'correct' if correct else 'incorrect'}.
                               User prompted if they are ready to continue.
                               """)

        elif action.type == ActionType.GENERATE_CODE:
            # payload: {"code": "...", "filename": "code.cpp"}
            code = p.get("code", "")
            fname = p.get("file_name", 'generated_code.out')

            ext = os.path.splitext(fname)[1].lstrip('.')
            lang = ext if ext else 'text'
            syntax = Syntax(code, lang, line_numbers=True)
            console.print(Panel(syntax, title=f"📝 Generated Code → {fname}"))

            save_to_file(code, fname)
            learner_fname = f"learner_{fname}"
            save_to_file(code, learner_fname)

            console.print()
            console.print(Panel(
                "[bold]Review the code above[/] in the console.\n"
                "When you're ready to address any TODO comments,\n"
                "[bold]press ENTER[/] to open your editor.\n"
                "When you are done editing, the file will be saved "
                "and the code will be compiled and evaluated.",
                title="Next Step",
                expand=False
            ))

            Prompt.ask("", default="", show_default=False)

            editor = os.environ.get('EDITOR', 'vi')
            subprocess.run([editor, learner_fname])

            with open(learner_fname) as f:
                final = f.read()
            console.print(Panel(
                Syntax(final, lang, line_numbers=True),
                title="🛠️ Your Edits",
                expand=False
            ))

            return Observation(result=f"""
                               User has saved their code file to {learner_fname} 
                               and the code is ready for compilation and 
                               evaluation.
                               """)

        elif action.type == ActionType.SYSTEM_CALL:
            cmd = action.payload
            console.print(Panel(f"Running shell command: {cmd}", title="Shell Command", expand=False))
            output = run_shell(cmd)
            return Observation(result=output)

        elif action.type == ActionType.REVIEW_FINISH:
            # payload: the reviewer's final review in string format
            # payload contains the "feed_back_summary" and the optional
            # "code_suggestions" keys

            feedback = p.get("feedback_summary", "")
            code_suggestions = p.get("code_suggestions", "")
            console.print(Panel(feedback, title="Review Summary", expand=False))
            # if code_suggestions then print it with text stating "Code Suggestions"
            if code_suggestions:
                console.print(Panel(code_suggestions, title="Code Suggestions", expand=False))
            return Observation(result="Review finished and feedback provided was the following:\n" + feedback)

        elif action.type == ActionType.QUERY_USER:
            # payload: {"question": "..."}
            question = p.get("question", "What would you like to do next?")
            console.print(Panel(question, title="User Query", expand=False))
            return Observation(result=f"User queried with question: {question}")

















