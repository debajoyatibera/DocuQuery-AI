import sys
from unittest.mock import MagicMock, patch

import pytest

from core.answer_generation import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationError,
    QwenLocalAnswerGenerator,
)
from core.context import BuiltContext


class FakeAnswerGenerator(AnswerGenerator):
    def generate(self, question: str, context: BuiltContext) -> GeneratedAnswer:
        if not question:
            raise ValueError("Question cannot be empty.")
        if question == "error_trigger":
            raise GenerationError("Simulated generation error.")

        # Return a simple deterministic response for testing
        return GeneratedAnswer(answer_text=f"Fake answer to: {question}")


def test_generated_answer_dataclass():
    answer = GeneratedAnswer(answer_text="This is an answer.")
    assert answer.answer_text == "This is an answer."


def test_generated_answer_allows_empty_text():
    # Empty answer text might happen if the generator refuses to answer or hits an empty return
    answer = GeneratedAnswer(answer_text="")
    assert answer.answer_text == ""


def test_answer_generator_is_abstract():
    with pytest.raises(TypeError):
        AnswerGenerator()


def test_fake_implementation_satisfies_contract():
    # Create empty context (we aren't testing ContextBuilder here, just the generator interface)
    context = BuiltContext(formatted_text="", used_chunks=[])

    generator = FakeAnswerGenerator()
    result = generator.generate("What is X?", context)

    assert isinstance(result, GeneratedAnswer)
    assert result.answer_text == "Fake answer to: What is X?"


def test_generation_error_is_exception():
    err = GenerationError("Test error")
    assert isinstance(err, Exception)
    assert str(err) == "Test error"


def test_generation_error_can_be_raised():
    context = BuiltContext(formatted_text="", used_chunks=[])
    generator = FakeAnswerGenerator()

    with pytest.raises(GenerationError, match="Simulated generation error."):
        generator.generate("error_trigger", context)


def test_qwen_generator_constructor_loads_model():
    mock_llama = MagicMock()
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
        generator = QwenLocalAnswerGenerator(
            model_path="dummy.gguf",
            n_ctx=2048,
            n_threads=4,
            temperature=0.2,
            max_tokens=256,
        )

        mock_llama.assert_called_once_with(
            model_path="dummy.gguf", n_ctx=2048, n_threads=4, verbose=False
        )
        assert generator.temperature == 0.2
        assert generator.max_tokens == 256


def test_qwen_generator_generate_success():
    mock_llama_instance = MagicMock()
    mock_llama_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Mocked answer"}}]
    }

    mock_llama = MagicMock(return_value=mock_llama_instance)
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
        generator = QwenLocalAnswerGenerator(model_path="dummy.gguf")

        context = BuiltContext(formatted_text="Mock context", used_chunks=[])
        result = generator.generate("Mock question", context)

        assert isinstance(result, GeneratedAnswer)
        assert result.answer_text == "Mocked answer"

        mock_llama_instance.create_chat_completion.assert_called_once()
        kwargs = mock_llama_instance.create_chat_completion.call_args.kwargs

        # Check params
        assert kwargs["temperature"] == generator.temperature
        assert kwargs["max_tokens"] == generator.max_tokens

        # Check prompt formatting constraints
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "You are an AI assistant" in messages[0]["content"]
        assert "Do not invent information" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        assert "Mock context" in messages[1]["content"]
        assert "Mock question" in messages[1]["content"]


def test_qwen_generator_load_error():
    mock_llama = MagicMock(side_effect=RuntimeError("Failed to load component"))
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
        with pytest.raises(
            GenerationError, match="Failed to load local Qwen model: Failed to load"
        ):
            QwenLocalAnswerGenerator(model_path="dummy.gguf")


def test_qwen_generator_generate_error():
    mock_llama_instance = MagicMock()
    mock_llama_instance.create_chat_completion.side_effect = ValueError(
        "Generation crashed internally"
    )

    mock_llama = MagicMock(return_value=mock_llama_instance)
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
        generator = QwenLocalAnswerGenerator(model_path="dummy.gguf")
        context = BuiltContext(formatted_text="Mock context", used_chunks=[])

        with pytest.raises(
            GenerationError,
            match="Local Qwen generation failed: Generation crashed internally",
        ):
            generator.generate("question", context)
