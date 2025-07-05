import os
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

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
            console.print(p["message"])
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

        elif action.type == ActionType.QUIZ_USER:
            console.log("[green]Quiz questions generated.[/]")
            return Observation(result="Quiz questions generated.")

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
