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


class QwenLocalAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            # Import lazily to avoid making llama-cpp-python a hard dependency
            # for module-level imports or systems without local inference setup.
            from llama_cpp import Llama

            self._model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
        except Exception as e:
            raise GenerationError(f"Failed to load local Qwen model: {str(e)}") from e

    def generate(
        self,
        question: str,
        context: BuiltContext,
    ) -> GeneratedAnswer:
        system_instruction = (
            "You are an AI assistant that answers questions using only the provided context. "
            "If the answer is not contained in the context, say that the provided context does "
            "not contain the answer. Do not invent information."
        )

        user_content = f"Context:\n{context.formatted_text}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self._model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer_text = response["choices"][0]["message"]["content"]
            return GeneratedAnswer(answer_text=answer_text)
        except Exception as e:
            raise GenerationError(f"Local Qwen generation failed: {str(e)}") from e
