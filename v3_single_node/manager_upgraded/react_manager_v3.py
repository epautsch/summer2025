import json
import re
import subprocess
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


######### ANSI Colors #########

YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

############## model and processor setup ############

model_id = "google/gemma-3-27b-it"
llm_model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
processor=AutoProcessor.from_pretrained(model_id)

############ utility funcs ##############

def strip_markdown_fences(text: str) -> str:
#    lines = text.strip().splitlines()
#    if len(lines) >= 2 and re.match(r"^```", lines[0]) and re.match(r"^```", lines[-1]):
#        return "\n".join(lines[1:-1]).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def run_shell(cmd: str) -> str:
    print(f"{BLUE}Running shell command:{RESET} {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    decoded = out.decode("utf-8", errors="ignore")
    print(f"{BLUE}Shell output:{RESET}\n{decoded}")
    return decoded

def save_to_file(text: str, filename: str):
    print(f"{GREEN}Saving to file:{RESET} {filename}")
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
    payload: Dict[str, Any]

@dataclass
class Observation:
    result: str

@dataclass
class HistoryManager:
    summarizer: Any
    history: List[str] = field(default_factory=list)
    word_limit: int = 400

    def add(self, entry: str):
        self.history.append(entry)
        if len(self.get_full().split()) > self.word_limit:
            summary = self.summarizer.generate(self.get_full())
            self.history = [f"History summary: {summary}"]

    def get_full(self) -> str:
        return "\n".join(self.history)

@dataclass
class Executor:
    def execute(self, action: Action) -> Observation:
        if action.type == ActionType.SYSTEM_CALL:
            if isinstance(action.payload, str):
                cmd = action.payload
            else:
                cmd = action.payload.get('cmd', '')
            output = run_shell(cmd)
            return Observation(result=output)

        elif action.type == ActionType.CODE:
            if isinstance(action.payload, dict):
                code = action.payload.get('input', '')
                fname = action.payload.get('filename', 'code.out')
            else:
                code = action.payload
                fname = 'code.out'
            save_to_file(code, fname)
            return Observation(result=f"Saved code to {fname}")

        elif action.type == ActionType.ANALYZE:
            return Observation(result="Analysis complete.")

        elif action.type == ActionType.FINISH:
            return Observation(result="Finished workflow.")

        else:
            return Observation(result=f"Unknown action: {action.type}")

@dataclass
class ManagerModel:
    model: Any
    processor: Any
    system_prompt: str
    max_new_tokens: int

    def generate(self, user_prompt: str) -> str:
        payload = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
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

@dataclass
class Agent:
    role_description: str
    model: ManagerModel
    history_mgr: HistoryManager
    thought: str = ''
    action: Action = None
    observation: str = ''

    def step(self, user_input: str = None):
        prompt_parts = [self.role_description, self.history_mgr.get_full()]
        if user_input:
            prompt_parts.append(f"User: {user_input}")
        else:
            prompt_parts.append(f"Observation: {self.observation}")
        prompt = "\n".join(filter(None, prompt_parts)).strip()

        while True:
            raw = self.model.generate(prompt)
            try:
                data = json.loads(strip_markdown_fences(raw))
                break
            except json.JSONDecodeError:
                print(f"{YELLOW}Warning:{RESET} Invalid JSON, retrying...\n{raw}\n")
                self.history_mgr.add(f"Observation: Manager JSON parse failed: {raw}")
                prompt += (
                    "\nYour last response was not valid JSON. "
                    "Please reply with only a JSON object containing keys 'thought', 'action', and 'payload'."
                )

        self.thought = data['thought']
        self.action = Action(type=ActionType[data['action'].upper()], payload=data['payload'])

        print(f"{YELLOW}Thought:{RESET} {self.thought}")
        print(f"{GREEN}Action:{RESET} {self.action}")
        print(f"{GREEN}Payload:{RESET} {self.action.payload}")

        self.history_mgr.add(f"Thought: {self.thought}")
        self.history_mgr.add(f"Action: {self.action.type.name} | Payload: {self.action.payload}")

def main_react():
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
    summarizer_system_prompt = (
        "You are a concise summarizer. Given a long conversation between a Manager "
        "and sub-agents, produce a short summary (100-150 words) that captures the key "
        "decisions (Thought, Action, Observation) *and* retains the original task context."
    )

    manager_llm = ManagerModel(
        model=llm_model,
        processor=processor,
        system_prompt=manager_system_prompt,
        max_new_tokens=2048
    )

    summarizer_llm = ManagerModel(
        model=llm_model,
        processor=processor,
        system_prompt=summarizer_system_prompt,
        max_new_tokens=512
    )

    history_mgr = HistoryManager(summarizer=summarizer_llm)

    executor = Executor()

    agent = Agent(
        role_description=manager_system_prompt,
        model=manager_llm,
        history_mgr=history_mgr
    )

    while True:
        task = input("Enter a task (or 'quit' to exit): ")
        if not task or task.strip().lower() in ('quit', 'exit'):
            print("Goodbye!")
            break
        
        agent.step(user_input=task)

        while agent.action.type is not ActionType.FINISH:
            obs = executor.execute(agent.action)
            agent.observation = obs.result
            print(f"{BLUE}Observation:{RESET} {agent.observation}")
            history_mgr.add(f"Observation: {agent.observation}")
            agent.step()

    print(f"{GREEN}==== Task complete ===={RESET}\n")

if __name__ == "__main__":
    main_react()

