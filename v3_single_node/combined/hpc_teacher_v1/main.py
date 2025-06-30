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


def main_react():
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

