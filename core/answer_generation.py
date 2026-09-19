from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.context import BuiltContext


class GenerationError(Exception):
    """Raised when the underlying LLM provider encounters an error."""

    pass


@dataclass
class GeneratedAnswer:
    answer_text: str


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        context: BuiltContext,
    ) -> GeneratedAnswer:
        """
        Generate an answer using the provided question and formatted context.
        """
        pass
