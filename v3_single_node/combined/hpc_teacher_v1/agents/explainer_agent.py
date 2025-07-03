from dataclasses import dataclass

from core.action import Action, ActionType
from agents.base_agent import BaseAgent


@dataclass
class ExplainerAgent(BaseAgent):
    """
    Agent that specializes in explaining concepts or answering follow-up questions.
    Maintains its own history and emits EXPLAIN_CONCEPT actions.
    """

    def explain_concept_action(self, concept: str) -> Action:
        """
        Request a clear explanation for the given concept, with examples if helpful.
        Emits EXPLAIN_CONCEPT with payload { explanation: str, examples: List[str] }.
        """
        prompt = (
            f"Explain the concept '{concept}' clearly, including examples if helpful. "
            "Return JSON: { 'action': 'EXPLAIN_CONCEPT', 'payload': "
            "{ 'explanation': <str>, 'examples': [ ... ] } }."
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.EXPLAIN_CONCEPT])
        return action

    def answer_question_action(self, concept: str, question: str) -> Action:
        """
        Provide a targeted answer to a follow-up question about the concept.
        Emits EXPLAIN_CONCEPT with payload { explanation: str, examples: List[str] }.
        """
        prompt = (
            f"The learner asks: '{question}' about the concept '{concept}'. "
            "Answer clearly and include examples if helpful. "
            "Return JSON: { 'action': 'EXPLAIN_CONCEPT', 'payload': "
            "{ 'explanation': <str>, 'examples': [ ... ] } }."
        )
        raw = self._generate(prompt)
        action = self._parse_action(raw, expect=[ActionType.EXPLAIN_CONCEPT])
        return action
