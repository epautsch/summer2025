import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from rich.console import Console
from rich.traceback import install

from core.model import LLMClient
from core.history_manager import HistoryManager
from core.executor import Executor
from agents.session_agent import SessionAgent, SessionState
from agents.lesson_planner_agent import LessonPlannerAgent
from agents.explainer_agent import ExplainerAgent
from prompts.lesson_planner_prompt import LESSON_PLANNER_PROMPT
from prompts.session_system_prompt import SESSION_SYSTEM_PROMPT
from prompts.explainer_prompt import EXPLAINER_PROMPT
from prompts.summarizer_prompt import SUMMARIZER_PROMPT

console = Console()
install()


def main():
    model_id = "google/gemma-3-27b-it"
    hf_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    session_llm = LLMClient(
        model=hf_model,
        processor=processor,
        system_prompt=SESSION_SYSTEM_PROMPT,
        max_new_tokens=2048
    )

    planner_llm = LLMClient(
        model=hf_model,
        processor=processor,
        system_prompt=LESSON_PLANNER_PROMPT,
        max_new_tokens=1024
    )

    explainer_llm = LLMClient(
        model=hf_model,
        processor=processor,
        system_prompt=EXPLAINER_PROMPT,
        max_new_tokens=1024
    )

    summarizer_llm = LLMClient(
        model=hf_model,
        processor=processor,
        system_prompt=SUMMARIZER_PROMPT,
        max_new_tokens=1024
    )

    session_history = HistoryManager(summarizer=summarizer_llm)
    planner_history = HistoryManager(summarizer=summarizer_llm)
    explainer_history = HistoryManager(summarizer=summarizer_llm)

    planner = LessonPlannerAgent(model=planner_llm, history=planner_history)
    explainer = ExplainerAgent(model=explainer_llm, history=explainer_history)

    executor = Executor()

    session = SessionAgent(
        model=session_llm,
        history=session_history,
        executor=executor,
        planner=planner,
        explainer=explainer,
    )

    session.run()


if __name__ == "__main__":
    main()
