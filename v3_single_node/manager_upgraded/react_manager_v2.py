import json
import subprocess
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re


###### model and processor ########

MODEL_ID = "google/gemma-3-27b-it"
print(f"[DEBUG] Loading model {MODEL_ID}...")
model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("[DEBUG] Model and processor ready.")

############## utility funcs #################

def strip_markdown_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and re.match(r"^```", lines[0]) and re.match(r"^```", lines[-1]):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()

def run_shell(cmd: str) -> str:
    print(f"DEBUG] Running shell command: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    decoded = out.decode("utf-8", errors="ignore")
    print(f"[DEBUG] Shell output:\n{decoded}")
    return decoded

def save_to_file(text: str, filename: str):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, "w") as f:
        f.write(text.strip() + "\n")
    print(f"[DEBUG] Saved file: {filename}")

def write_code_files(json_str: str) -> list:
    data = json.loads(strip_markdown_fences(json_str))
    if "files" not in data:
        raise ValueError("CODE action must return a JSON object with a 'files' list")
    filenames = []
    for file in data["files"]:
        fname = file.get("filename") or file.get("name")
        content = file.get("content")
        if fname and content is not None:
            save_to_file(content, fname)
            filenames.append(fname)
    print(f"[DEBUG] Generated code files: {filenames}")
    return filenames

############## calling the llm #################

MANAGER_SYSTEM_PROMPT = '''
You are an HPC manager AI. You orchestrate end-to-end HPC tasks across any paradigm (CUDA, SYCL, OpenMP, MPI, etc.) by choosing exactly one of these actions each turn:

1) SPEC:
    Generate a concise software specification (≤200 words, ≤10 bullet points) from the original task and history.
    Payload: the spec text.

2) CODE:
    Using your spec, produce a JSON object listing one or more files:
    { "files": [ {"filename": "...", "content": "..."}, ...] }
    Payload: that json as a string.

3) INVESTIGATE:
    Run a shell command via run_shell to inspect the system (e.g., check compiler versions, list files). You don't need to call run_shell, you only need to provide the command that will run in the shell itself.
    Payload: the shell command to execute.

4) COMPILE:
    Compile a build via run_shell.
    Payload: the compile command.

5) RUN:
    Run an executable via run_shell.
    Payload: the run command.

6) SUMMARIZE:
    If your history grows greater than 300 words, condense the entire history into a 50-100 word summary, preserving the task context.
    Payload: the summary text.

7) FINISH:
    Provide a final summary of the process and list of generated files.
    Payload: your final summary text.

On each turn, read the histroy and choose the best action. Return a raw JSON object:
    { "action": "SPEC|CODE|INVESTIGATE|COMPILE|RUN|SUMMARIZE|FINISH", "payload": "..." }
No extra explanation or markdown fences.
'''

def call_llm(prompt: str, content: str, max_new_tokens: int = 2048) -> str:
    print("[DEBUG] Invoking LLM...")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "text", "text": content}]}
    ]
    raw = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = raw["input_ids"].shape[-1]
    with torch.inference_mode():
        gen = model.generate(
            **raw, max_new_tokens=max_new_tokens, do_sample=False
        )
    output_ids = gen[0][input_len:]
    decoded = processor.decode(output_ids, skip_special_tokens=True)
    print(f"[DEBUG] LLM replied (truncated): {decoded[:200]}...")
    return strip_markdown_fences(decoded)

def parse_manager_response(resp: str) -> dict:
    try:
        obj = json.loads(resp)
        print(f"[DEBUG] Parsed action: {obj.get('action')}")
        return obj
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse manager JSON: {e}")
        print(f"[ERROR] Raw response: {resp}")
        raise

############## manager's react loop ###################

def main_react():
    original_task = "Implement basic matrix multiplication in CUDA."
    history = f"Original Task: {original_task}"
    generated_files = []

    for iteration in range(20):
        print(f"\n=========== Loop iteration {iteration} ===========")
        
        word_count = len(history.split())
        print(f"[DEBUG] History word count: {word_count}")
        if word_count > 300:
            print("[DEBUG] Triggering SUMMARIZE action...")
            resp = call_llm(MANAGER_SYSTEM_PROMPT, history)
            mgr = parse_manager_response(resp)
            if mgr.get("action") == "SUMMARIZE":
                history = f"Original Task: {original_task}\nHistory Summary:\n{mgr['payload']}"
                continue

        resp = call_llm(MANAGER_SYSTEM_PROMPT, history)
        mgr = parse_manager_response(resp)
        action = mgr.get("action")
        payload = mgr.get("payload", "")
        print(f"[DEBUG] Manager decided: {action}")
        history += f"\n\n[Manager {iteration}] Action: {action}, Payload: {payload}"

        if action == "SPEC":
            save_to_file(payload, "spec.txt")

        elif action == "CODE":
            new_files = write_code_files(payload)
            generated_files.extend(new_files)

        elif action == "INVESTIGATE":
            print(f"[DEBUG] INVESTIGATE command: {payload}")
            out = run_shell(payload)
            history += f"\n[Investigate Output]\n{out}"

        elif action == "COMPILE":
            print(f"[DEBUG] COMPILE command: {payload}")
            out = run_shell(payload)
            history += f"\n[Compile Output]\n{out}"

        elif action == "RUN":
            print(f"[DEBUG] RUN command: {payload}")
            out = run_shell(payload)
            history += f"\n[Run Output]\n{out}"

        elif action == "FINISH":
            print("=== FINAL SUMMARY ===")
            print(payload)
            print("Generated files:", generated_files)
            return

        else:
            raise RuntimeError(f"Unknown action: {action}")

    print("Reached maximum iteration limit without 'finish' action.")

if __name__ == "__main__":
    main_react()

