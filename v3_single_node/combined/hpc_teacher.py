import json
import re
import subprocess
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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
from rich.prompt import Prompt


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
processor=AutoProcessor.from_pretrained(model_id)

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

    def show_history(self):
        table = Table(title="Agent History")
        table.add_column("Step", style="dim", width=6, justify="right")
        table.add_column("Entry")
        for i, entry in enumerate(self.history, 1):
            table.add_row(str(i), entry)
        console.print(table)

@dataclass
class Executor:
    def execute(self, action: Action) -> Observation:
        console.log(f"[bold cyan]Executing action[/] → {action.type.name}")
        if action.type == ActionType.SYSTEM_CALL:
            if isinstance(action.payload, str):
                cmd = action.payload
            else:
                cmd = action.payload.get('cmd', '')
            output = run_shell(cmd)
            return Observation(result=output)

        elif action.type == ActionType.CODE:
            code = action.payload.get('input', '')
            fname = action.payload.get('filename', 'code.out')
            ext = os.path.splitext(fname)[1].lstrip('.')
            lang = ext if ext else 'text'
            syntax = Syntax(code, lang, line_numbers=True)
            console.print(Panel(syntax, title=f"Generated Code → {fname}"))
            save_to_file(code, fname)
            return Observation(result=f"Saved code to {fname}")

        elif action.type == ActionType.ANALYZE:
            console.log("[green]Analysis complete.[/]")
            return Observation(result="Analysis complete.")

        elif action.type == ActionType.FINISH:
            console.log(f"[bold magenta]Workflow finished:[/] {action.payload}")
            return Observation(result="Finished workflow.")

        else:
            console.log(f"[red]Unknown action:[/] {action.type}")
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
        with console.status("Generating response...", spinner="dots"):
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
class LessonPlanner:
    llm: ManagerModel
    default_topics: List[str] = field(default_factory=lambda: [
        "Introduction to Parallel Computing Concepts (shared memory vs. distributed memory)",
        "OpenMP: Parallelizing Loops with Directives",
        "CUDA: Vector Addition with Thrust",
        "MPI: Hello World and Basic Point-to-Point Communication",
        "SYCL: Simple Kernel for Array Multiplication"
    ])

    def suggest_topics(self) -> List[str]:
        prompt = (
            "List 5 beginner-friendly HPC topics across different backends (CUDA, OpenMP, "
            "MPI, SYCL, etc.), each on its own line."
        )
        resp = self.llm.generate(prompt)
        topics = [t.strip() for t in resp.splitlines() if t.strip()]
        return topics or self.default_topics

    def create_outline(self, topic: str) -> List[str]:
        prompt = (
            f"Create a 5-step teaching outline for the topic: {topic}, including "
            "conceptual and hands-on coding steps."
        )
        resp = self.llm.generate(prompt)
        return [line.strip() for line in resp.splitlines() if line.strip()]

@dataclass
class Quizzer:
    llm: ManagerModel

    def generate_questions(self, concept: str) -> List[Dict[str, str]]:
        prompt = (
            f"Write 3 short-answer questions about '{concept}' and return JSON list of {{'q':..., 'a':...}} entries."
        )
        raw = self.llm.generate(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            console.print("[red]Failed to parse quiz JSON.[/]")
            return []

    def grade_answer(self, correct: str, user_ans: str) -> bool:
        return user_ans.strip().lower() == correct.strip().lower()

@dataclass
class CodeEvaluator:
    compile_cmd: str = ""
    run_cmd: str = ""

    def set_backend(self, backend: str, ext: str):
        if backend.lower() == "cuda":
            self.compile_cmd = f"nvcc -o {{out}} {{src}}"
            self.run_cmd = "./{out}"
        elif backend.lower() == "sycl":
            self.compile_cmd = f"dpcpp -o {{out}} {{src}}"
            self.run_cmd = "./{out}"
        elif backend.lower() == "openmp":
            self.compile_cmd = f"gcc -fopenmp -o {{out}} {{src}}"
            self.run_cmd = "./{out}"
        elif backend.lower() == "mpi":
            self.compile_cmd = f"mpicc -o {{out}} {{src}}"
            self.run_cmd = "mpirun -n 4 ./{{out}}"
        else:
            self.compile_cmd = f"g++ -std=c++17 -o {{out}} {{src}}"
            self.run_cmd = "./{out}"

    def compile(self, src: str, out: str) -> str:
        cmd = self.compile_cmd.format(src=src, out=out)
        return run_shell(cmd)

    def run(self, out: str) -> str:
        cmd = self.run_cmd.format(out=out)
        return run_shell(cmd)

@dataclass
class CodeTutor:
    llm: ManagerModel
    evaluator: CodeEvaluator
    ext: str = ""

    def generate_skeleton(self, task: str, backend: str) -> str:
        prompt = (
            f"Generate a {self.ext} source file for '{task}' using the {backend} backend. "
            "Include three TODO comments where the student should implement key parts. Return only "
            "the code."
        )
        return self.llm.generate(prompt)

    def evaluate_submission(self, src_file: str, exe: str, expected_output: str) -> Tuple[bool, str]:
        compile_log = self.evaluator.compile(src_file, exe)
        if 'error' in compile_log.lower():
            return False, compile_log
        output = self.evaluator.run(exe).strip()
        return (output == expected_output, output)

@dataclass
class SessionManager:
    planner: LessonPlanner
    quizzer: Quizzer
    tutor: CodeTutor
    history: HistoryManager

    def run(self):
        console.print("[bold green]👋 Welcome to the HPC Tutor![/]")

        topics = self.planner.suggest_topics()
        for i, t in enumerate(topics, 1): console.print(f"{i}. {t}")
        choice = Prompt.ask("Choose a topic by number or type a new one")
        try:
            topic = topics[int(choice) - 1]
        except Exception:
            topic = choice

        backend = Prompt.ask(
            "Select implementation backend [cuda, sycl, openmp, mpi, cpp]", default="cpp"
        )
        ext_map = {"cuda":"cu", "sycl":"cpp", "openmp":"c", "mpi":"c", "cpp":"cpp"}
        ext = ext_map.get(backend.lower(), "cpp")
        self.tutor.ext = ext
        self.tutor.evaluator.set_backend(backend, ext)

        outline = self.planner.create_outline(topic)
        console.print(Panel("\n".join(outline), title=f"Lesson Outline: {topic} ({backend})"))

        for step in outline:
            console.print(Rule(f"📖 {step}"))

            explanation = self.planner.llm.generate(f"Explain the concept: {step}")
            console.print(Panel(explanation, title="Concept Explanation"))

            questions = self.quizzer.generate_questions(step)
            for qa in questions:
                user_ans = Prompt.ask(qa['q'])
                if self.quizzer.grade_answer(qa['a'], user_ans):
                    console.print("[green]✔ Correct![/]")
                else:
                    console.print(f"[red]✖ Incorrect.[/] Expected: {qa['a']}")

            skeleton = self.tutor.generate_skeleton(f"{topic}: {step}", backend)
            filename = f"lesson_step_{outline.index(step)+1}.{ext}"
            console.print(Panel(Syntax(skeleton, ext, line_numbers=True), title=f"Code Skeleton → {filename}"))
            save_to_file(skeleton, filename)

            console.print("Paste your completed code below (end with empty line):")
            user_lines = []
            while True:
                line = input()
                if not line.strip(): break
                user_lines.append(line)
            completed = "\n".join(user_lines)
            src_file = f"completed_step_{outline.index(step)+1}.{ext}"
            save_to_file(completed, src_file)

            ok, result = self.tutor.evaluate_submission(src_file, "lesson_exec", expected_output="EXPECTED_OUTPUT")
            if ok:
                console.print("[bold green]🎉 Code ran successfully![/]")
            else:
                console.print(Panel(result, title="Errors / Output"))

        console.print(Rule("🏁 Lesson Complete!"))
        summary = self.planner.llm.generate(f"Summarize the lesson on {topic} using {backend} and key takeaways.")
        console.print(Panel(summary, title="Lesson Summary"))
        hw = self.planner.llm.generate(f"Generate 4 homework exercises for {topic} on {backend}.")
        console.print(Panel(hw, title="Homework"))


def main_react():
    manager_system_prompt = (
        "You are an HPC tutor agent responsible for teaching high-perforance computing (HPC) concepts "
        "interactively by guiding a user through lessons, providing quizzes, evaluating answers, assigning "
        "homework, testing user-written code, and demonstrating HPC concepts through code examples. Your "
        "interactions must help users progressively master HPC-related skills, including parallel computing, "
        "algorithms, performance optimization, system interactions and more. Each learning session should take "
        "on a similar format: a user chooses from a list of pre-written learning topics or tells you a new one "
        "they would like to learn; you explain the concept(s) of the learning topic to the user through paragraph "
        "explanations or code examples; you quiz the users knowledge of the topic you just explained to them "
        "through a few multiplie-choice questions; you generate code with TODO items for the user to complete; "
        "you compile and run the user's code to check for correctness; you provide the user with feedback on their "
        "code; you provide the user with homework exerises based on how the learning session went."
        "When provided a 'Thought:' prompt, you must reply in JSON with exactly three fields: "
        "\"thought\", \"action\", \"payload\". Valid actions are :\n"
        "1) CREATE_LESSON_PLAN\n"
        "2) EXPLAIN_CONCEPT\n"
        "3) QUIZ_USER\n"
        "4) CODE\n"
        "5) SYSTEM_CALL\n"
        "6) GENERATE_HOMEWORK\n"
        "7) FINISH\n"
        "You must use each action precisely as described below:\n"
        " • CREATE_LESSON_PLAN\n"
        "   • Generate a detailed lesson plan based on either a predefined topic or a user-suggested topic.\n"
        "   • The lesson plan should clearly outline the topics and their correspnding objectives.\n"
        "   • payload = {\"topic\": \"<topic_name>\", \"objectives\": [\"objective_1\", \"objective_2, ...]}\n"
        " • EXPLAIN_CONCEPT\n"
        "   • Clearly explain HPC concepts using a blend of descriptive paragraphs, illustrative code snippets, "
        "and relatable analogies to enhance understanding.\n"
        "   • payload = {\"explanation\": \"<clear_and_detailed_explanation>\", \"examples\": [\"example_snippet_"
        "or_analogy_1\", \"example_snippet_or_analogy_2\", ...]}\n"
        " • QUIZ_USER\n"
        "   • Test the user's understanding of the concepts just explained through carefully crafted multiple-"
        "choice questions.\n"
        "   • Track user responses internally:\n"
        "       • If the user correctly answers three questions in a row, consider the concept understood.\n"
        "       • If the user continues to answer incorrectly, keep asking questions.\n"
        "   • payload = {\"question\": \"<quiz_question>\", \"choices\": [\"choice_a\", \"choice_b\", "
        "\"choice_c\", \"choice_d\"], \"correct_answer\": \"<correct_choice_letter>\"}\n"
        " • CODE\n"
        "   • Generate clear, instructional code examples with clearly indicated TODO sections for the user to "
        "complete, reinforcing practical coding skills.\n"
        "   • Avoid requesting terminal-based user inputs unless explicitly instructed.\n"
        "   • payload = {\"description\": \"<instructions_or_explanation>\", \"filename\": \"<appropriate_filename"
        ".ext>\", \"code\": \"<code_with_todos>\"}\n"
        " • SYSTEM_CALL\n"
        "   • Execute necessary system commands, including compiling and running code or retrieving system "
        "information.\n"
        "   • Execute only one command at a time. Do not chain commands.\n"
        "   • payload = \"<single_shell_command>\"\n"
        " • GENERATE_HOMEWORK\n"
        "   • Assign homework tasks designed to reinforce the lesson's key concepts and address the user's "
        "performance in quizzes and coding tasks.\n"
        "   • Tasks should be concise yet effective, neither simple multiple-choice questions nor extensive "
        "coding projects.\n"
        "   • payload = {\"tasks\": [\"task_1_description\", \"task_2_description\", ...]}\n"
        " • FINISH\n"
        "   • Use this action to conclude the learning session clearly and comprehensively.\n"
        "   • Provide a summary of what concepts were covered, how the user performed in quizzes and coding "
        "tasks, key insights gained, and suggested areas for future improvement or practice.\n"
        "   • payload = \"<summary_of_session_and_user_performance>\"\n"
        "Additional Guidlines and Requirements:\n"
        "   • Always structure explanations to build upon previous concepts, ensuring clarity and depth.\n"
        "   • Be responsive to the user's learning pace, adjusting complexity and detail as needed.\n"
        "   • Provide specific, constructive feedback on quizzes and coding exercises, clearly indicating "
        "areas of strength and needed improvement.\n"
        "   • Maintain an interactive and engaging teaching style to maximize user engagement and knowledge "
        "retention.\n"
        "Always adhere to the provided action formats exactly. Your goal is to build deep HPC understanding and "
        "practical proficiency interactively."
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

    planner = LessonPlanner(llm=manager_llm)
    quizzer = Quizzer(llm=manager_llm)
    evaluator = CodeEvaluator()
    tutor = CodeTutor(llm=manager_llm, evaluator=evaluator)
    session = SessionManager(planner=planner, quizzer=quizzer, tutor=tutor, history=history_mgr)

    session.run()

if __name__ == "__main__":
    main_react()

