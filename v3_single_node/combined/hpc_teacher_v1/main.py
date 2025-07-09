import argparse
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer
from rich.console import Console
from rich.traceback import install

from core.model import LLMClient
from core.history_manager import HistoryManager
from core.executor import Executor
from agents.session_agent import SessionAgent
from agents.explainer_agent import ExplainerAgent
from agents.quizzer_agent import QuizzerAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from prompts.session_system_prompt import SESSION_SYSTEM_PROMPT
from prompts.explainer_prompt import EXPLAINER_PROMPT
from prompts.quizzer_prompt import QUIZZER_PROMPT
from prompts.coder_prompt import CODER_PROMPT
from prompts.reviewer_prompt import REVIEWER_PROMPT
from prompts.summarizer_prompt import SUMMARIZER_PROMPT

console = Console()
install()


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch the HPC Tutor with a chosen model size"
    )
    p.add_argument(
        "--model-size",
        choices=["1b", "27b"],
        default="27b",
        help="Select the model size to use for the session."
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_id = f"google/gemma-3-{args.model_size}-it"
    console.print(f"[bold green]Using model:[/bold green] {model_id}")

    hf_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except OSError:
        processor = AutoTokenizer.from_pretrained(model_id)

    session_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=SESSION_SYSTEM_PROMPT,
        max_new_tokens=2048
    )

    explainer_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=EXPLAINER_PROMPT,
        max_new_tokens=1024
    )

    quizzer_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=QUIZZER_PROMPT,
        max_new_tokens=1024
    )

    coder_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=CODER_PROMPT,
        max_new_tokens=2048
    )

    summarizer_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=SUMMARIZER_PROMPT,
        max_new_tokens=1024
    )

    reviewer_llm = LLMClient(
        hf_model=hf_model,
        processor=processor,
        system_prompt=REVIEWER_PROMPT,
        max_new_tokens=1024
    )

    session_history = HistoryManager(summarizer=summarizer_llm)
    explainer_history = HistoryManager(summarizer=summarizer_llm)
    quizzer_history = HistoryManager(summarizer=summarizer_llm)
    coder_history = HistoryManager(summarizer=summarizer_llm)
    reviewer_history = HistoryManager(summarizer=summarizer_llm)

    explainer = ExplainerAgent(model=explainer_llm, history=explainer_history)
    quizzer = QuizzerAgent(model=quizzer_llm, history=quizzer_history)
    coder = CoderAgent(model=coder_llm, history=coder_history)
    reviewer = ReviewerAgent(model=reviewer_llm, history=reviewer_history)

    executor = Executor()

    session = SessionAgent(
        model=session_llm,
        history=session_history,
        executor=executor,
        explainer=explainer,
        quizzer=quizzer,
        coder=coder,
        reviewer=reviewer,
    )

    session.run()


if __name__ == "__main__":
    main()
