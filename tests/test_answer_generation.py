import pytest

from core.answer_generation import AnswerGenerator, GeneratedAnswer, GenerationError
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
