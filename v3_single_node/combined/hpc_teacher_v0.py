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
    CREATE_LESSON_PLAN = auto()
    EXPLAIN_CONCEPT = auto()
    QUIZ_USER = auto()
    CODE = auto()
    SYSTEM_CALL = auto()
    GENERATE_HOMEWORK = auto()
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
    word_limit: int = 800

    def add(self, entry: str):
        self.history.append(entry)
        
    def get_full(self) -> str:
        full_text = "\n".join(self.history)

        if len(full_text.split()) > self.word_limit:
           # console.print("[red][DEBUG SUMMARIZER][/]")
            summary = self.summarizer.generate(full_text)
           # console.print("[red][DEBUG SUMMARIZER END][/]\n")
            self.history = [f"History summary: {summary}"]
            return self.history[0]
        return full_text

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
        p = action.payload

        if action.type == ActionType.CREATE_LESSON_PLAN:
            # payload: {"topic": ..., "objectives": [...[}
            title = f"Lesson Plan: {p['topic']}"
            body = "\n".join(f"- {o}" for o in p["objectives"])
            console.print(Panel(body, title=title))
            return Observation(result="Displayed leeson plan.")

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
            concole.log("[green]Quiz questions generated.[/]")
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

        elif action.type == ActionType.GENERATE_HOMEWORK:
            console.log("[green]Homework generated.[/]")
            return Observation(result="Homework generated.")

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
        # DEBUG print
       # console.print(f"[red][DEBUG PAYLOAD][/]\n{payload}")
       # console.print(f"[red][DEBUG PAYLOAD END][/]\n")

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
           # console.print("[red][DEBUG MANAGERMODEL GEN][/]")
           # console.print(decoded)
           # console.print("[red][DEBUG MANAGERMODEL GEN END][/]\n")
            return decoded

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
    lesson_topic: str = ""
    lesson_objectives: List[str] = field(default_factory=list)

    def set_plan(self, topic: str, objectives: List[str]) -> None:
        self.lesson_topic = topic
        self.lesson_objectives = objectives

     # MARKED for removal
    """
    def suggest_topics(self) -> List[str]:
        prompt = (
            "List 5 beginner-friendly HPC topics across different backends (CUDA, OpenMP, "
            "MPI, SYCL, etc.), each on its own line."
        )
        resp = self.llm.generate(prompt)
        topics = [t.strip() for t in resp.splitlines() if t.strip()]
        return topics or self.default_topics
    """

    def create_lesson_plan(self, topic: str) -> List[str]:
        return (
            f"The user has chosen to learn about {topic}. "
            "Create the lesson plan as JSON with keys 'thought', 'action', and 'payload'."
        )

    def explain_concept(self, concept: str) -> str:
        return f"Explain the concept: {concept}"

    def answer_question(self, concept: str, question: str) -> str:
        return (
            f"The learner is asking a follow-up about '{concept}':\n"
            f"\"{question}\"\n"
            "Please answer clearly and concisely."
        )

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
    executor: Executor
    history: HistoryManager

    def _call_llm(self, user_prompt: str,
                  schema_hint: str = "Your last response was not valid JSON. Please reply with valid JSON.",
                  max_retries: int = 3) -> dict:

        hist = self.history.get_full()
        prefix = (hist + "\n") if hist else ""

        full_prompt = prefix + user_prompt
        last_err = None

        for attempt in range(1, max_retries + 1):
            raw = self.planner.llm.generate(full_prompt)
            self.history.add(f"UserPrompt (JSON): {user_prompt}")
            self.history.add(f"LLMResponse (raw JSON): {raw}")

            try:
                data = json.loads(strip_markdown_fences(raw))
                return data
            except json.JSONDecodeError as e:
                last_err = e
                console.print(f"[red]Warning:[/] JSON parse failed attempt {attempt}): {e}")
                full_prompt += f"\n{schema_hint}"

        raise last_err

    def _extract_explanation_data(self, raw: str) -> Tuple[str, List[str]]:
        try:
            payload = raw.get("payload", {})
            explanation = payload.get("explanation", raw)
            examples = payload.get("examples", [])
            return explanation, examples
        except Exception:
            return raw, []

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
        console.print(f"[red][DEBUG TOPIC CHOICE END][/]\n")
        
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
            
            """
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
            """
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
        "and sub-agents, produce a short summary (200-250 words) that captures the key "
        "decisions (Thought, Action, Observation) *and* retains the original task context."
    )

    manager_llm = ManagerModel(
        model=llm_model,
        processor=processor,
        system_prompt=manager_system_prompt,
        max_new_tokens=8192
    )

    summarizer_llm = ManagerModel(
        model=llm_model,
        processor=processor,
        system_prompt=summarizer_system_prompt,
        max_new_tokens=1024
    )
    history_mgr = HistoryManager(summarizer=summarizer_llm)
    executor = Executor()

    planner = LessonPlanner(llm=manager_llm)
    quizzer = Quizzer(llm=manager_llm)
    evaluator = CodeEvaluator()
    tutor = CodeTutor(llm=manager_llm, evaluator=evaluator)
    session = SessionManager(planner=planner, quizzer=quizzer, tutor=tutor, executor=executor, history=history_mgr)

    session.run()

if __name__ == "__main__":
    main_react()

