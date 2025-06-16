import json
import re
import subprocess
import os
from enum import Enum, auto
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from rich.console import Console
from rich.traceback import install
from rich.status import Status
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from mcp import MCPClient, ToolDefinition


######### rich stuff ##########

console = Console()
install()

############## model and processor setup ############

model_id = "google/gemma-3-27b-it"
llm_model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_id)

############ utility funcs ##############

def strip_markdown_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def run_shell(cmd: str) -> str:
    with console.status(f"⏳ [bold blue]Running shell command:[/] {cmd}", spinner="dots"):
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        out, _ = proc.communicate()
    text = out.decode("utf-8", errors="ignore")
    console.log(f"🔹 [bold blue]Shell output:[/]\n{text}")
    return text

def save_to_file(text: str, filename: str):
    console.log(f"💾 Saving to file: [bold green]{filename}[/]")
    with open(filename, "w") as f:
        f.write(text.strip() + "\n")

class ActionType(Enum):
    ANALYZE = auto()
    CODE = auto()
    SYSTEM_CALL = auto()
    FINISH = auto()

@dataclass
class Action:
    type: ActionType
    payload: dict

@dataclass
class Observation:
    result: str

@dataclass
class Executor:
    def execute(self, action: Action) -> Observation:
        console.log(f"[bold cyan]Executing action[/] → {action.type.name}")
        if action.type == ActionType.SYSTEM_CALL:
            cmd = action.payload.get('cmd', '')
            return Observation(result=run_shell(cmd))

        elif action.type == ActionType.CODE:
            code = action.payload.get('input', '')
            fname = action.payload.get('filename', 'code.out')
            lang = os.path.splitext(fname)[1].lstrip('.') or 'text'
            console.print(Panel(Syntax(code, lang, line_numbers=True), title=f"Generated Code → {fname}"))
            save_to_file(code, fname)
            return Observation(result=f"Saved code to {fname}")

        elif action.type == ActionType.ANALYZE:
            console.log("[green]Analysis complete.[/]")
            return Observation(result="Analysis complete.")

        elif action.type == ActionType.FINISH:
            console.log(f"[bold magenta]Workflow finished:[/] {action.payload.get('result')}" )
            return Observation(result=action.payload.get('result', ''))

        else:
            console.log(f"[red]Unknown action:[/] {action.type}")
            return Observation(result=f"Unknown action: {action.type}")

@dataclass
class ManagerModel:
    model: Any
    processor: Any
    system_prompt: str
    max_new_tokens: int

    def generate(self, prompt: str) -> str:
        payload = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        raw = self.processor.apply_chat_template(
            payload,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = raw["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(
                **raw,
                max_new_tokens=self.max_new_tokens,
                #cache_implementation="offloaded",
                do_sample=False
            )
        gen_ids = out[0][input_len:]
        decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
        return strip_markdown_fences(decoded)

############## Instantiate models ##########

manager_system_prompt = (
    "You are an HPC manager agent. Use JSON-RPC calls to 'analyze', 'code', 'system_call', or 'finish'."
)

manager_llm = ManagerModel(
    model=llm_model,
    processor=processor,
    system_prompt=manager_system_prompt,
    max_new_tokens=2048
)

################# MCP tools ###################

toold = [
    ToolDefinition(
        name="analyze",
        description="Internal planning or specification",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"]
        }
        ),
    ToolDefinition(
        name="code",
        description="Generate source code and save to file",
        parameters={
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "filename": {"type": "string"}
            },
            "required": ["input", "filename"]
        }
    ),
    ToolDefinition(
        name="system_call",
        description="Execute a shell command and return output",
        parameters={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"]
        }
    ),
    ToolDefinition(
        name="finish",
        description="Finish workflow with summary",
        parameters={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"]
        }
    )
]

######### MCP client ########

def llm_adapter(prompt: str) -> str:
    return manager_llm.generate(prompt)

mcp_client = MCPClient(llm_adapter=llm_adapter, tools=tools)
executor = Executor()

########### main lloop ########

def main():
    """
    manager_system_prompt = (
        "You are an HPC manager agent responsible for taking a user task and going through "
        "the end-to-end process to accomplish that task. Tasks could include (but are not "
        "limited to): software creation, answering questions about the system based on "
        "system calls you make, and more."
        "When given a 'Thought:' prompt, you must reply in JSON with exactly three fields: "
        "\"thought\", \"action\", \"payload\". Valid actions are :\n"
        "1) ANALYZE\n"
        "2) CODE\n"
        "3) SYSTEM_CALL\n"
        "4) FINISH\n"
        "Use each action as follow:\n"
        " • ANALYZE     - internal planning or specification (payload = {\"input\": <string>})\n"
        " • CODE        - generate source code (payload = {\"input\": <string> \"filename\": <string.ext>})\n"
        " • SYSTEM_CALL - execute shell commands (compile, run, inspect; payload = <command>)\n"
        " • FINISH      - terminate the workflow with a summary (payload = <string>)\n\n"
        "When you are finished with your task, ensure you use the FINISH action. Do not wait for "
        "another task from the user. "
        "When you are writing code, make sure the code does not try to get user input from the terminal "
        "unless the user specifically asks you to write the code that way. "
        "When using the SYSTEM_CALL action to run shell commands, run one command at a time. "
        "Do NOT chain commands with '&&'. Run one command, observe the output, and then continue. "
        "Do NOT wrap your JSON in markdown fences or include extra keys/text."
    )
    """
    console.print("[bold green]Starting MCP-based HPC Manager Agent[/]")
    while True:
        task = console.input("[bold]Enter a task[/] (or 'quit' to exit): ")
        if not task or task.strip().lower() in ('quit', 'exit'):
            console.print("[bold red]Goodbye![/]")
            break

        response = mcp_client.step(task)
        while True:
            method = response.method
            params = response.params
            if method == "finish":
                summary = params.get("result", "")
                console.print(Rule("[bold green]Workflow Summary"))
                console.print(Panel(summary))
                break

            action = Action(type=ActionType[method.upper()], payload=params)
            obs = executor.execute(action)

            response = mcp_client.send_observation(obs.result)

        console.print(Rule("[bold green]==== Task complete ===="))
        
if __name__ == "__main__":
    main()

