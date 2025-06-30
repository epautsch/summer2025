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

