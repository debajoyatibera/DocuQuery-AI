from typing import List

import pytest

from core.answer_generation import AnswerGenerator, GeneratedAnswer, GenerationError
from core.context import BuiltContext, ContextBuilder
from core.orchestration import (
    NO_EVIDENCE_RESPONSE,
    OrchestrationResult,
    RAGOrchestrator,
)
from core.retrieval import RetrievedChunk, Retriever


class FakeRetriever(Retriever):
    def __init__(self, chunks: List[RetrievedChunk] = None):
        self.chunks = chunks or []
        self.received_query = None

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        self.received_query = query
        if query == "trigger_retrieval_error":
            raise ValueError("Retrieval failed")
        return self.chunks


class FakeContextBuilder(ContextBuilder):
    def __init__(self, context: BuiltContext = None):
        self.context = context or BuiltContext(formatted_text="", used_chunks=[])
        self.received_chunks = None

    def build(
        self, chunks: List[RetrievedChunk], max_chars: int = 10000
    ) -> BuiltContext:
        self.received_chunks = chunks
        if chunks and chunks[0].text == "trigger_context_error":
            raise ValueError("Context building failed")
        return self.context


class FakeAnswerGenerator(AnswerGenerator):
    def __init__(self, answer: GeneratedAnswer = None):
        self.answer = answer or GeneratedAnswer(answer_text="Fake answer")
        self.received_question = None
        self.received_context = None

    def generate(self, question: str, context: BuiltContext) -> GeneratedAnswer:
        self.received_question = question
        self.received_context = context
        if question == "trigger_generation_error":
            raise GenerationError("Generation failed")
        return self.answer


def create_dummy_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )


def test_successful_orchestration():
    chunk1 = create_dummy_chunk("Raw chunk 1")
    chunk2 = create_dummy_chunk("Raw chunk 2")

    # Context builder only uses chunk1
    built_context = BuiltContext(
        formatted_text="Formatted chunk 1", used_chunks=[chunk1]
    )

    retriever = FakeRetriever(chunks=[chunk1, chunk2])
    builder = FakeContextBuilder(context=built_context)
    generator = FakeAnswerGenerator(answer=GeneratedAnswer(answer_text="Final Answer"))

    orchestrator = RAGOrchestrator(
        retriever=retriever, context_builder=builder, answer_generator=generator
    )

    result = orchestrator.answer("What is X?")

    # Verify inputs received by components
    assert retriever.received_query == "What is X?"
    assert builder.received_chunks == [chunk1, chunk2]
    assert generator.received_question == "What is X?"
    assert generator.received_context == built_context

    # Verify exact result
    assert result.answer_text == "Final Answer"
    # Verify we return ONLY context-builder-selected evidence, not all retrieved chunks
    assert result.evidence == [chunk1]


def test_empty_evidence_short_circuits_generation():
    retriever = FakeRetriever(chunks=[])
    builder = FakeContextBuilder(
        context=BuiltContext(formatted_text="", used_chunks=[])
    )
    generator = FakeAnswerGenerator()

    orchestrator = RAGOrchestrator(
        retriever=retriever, context_builder=builder, answer_generator=generator
    )

    result = orchestrator.answer("What is X?")

    # Generator should NOT be called
    assert generator.received_question is None

    # Verify controlled response and empty evidence
    assert result.answer_text == NO_EVIDENCE_RESPONSE
    assert result.evidence == []


def test_whitespace_question_validation():
    orchestrator = RAGOrchestrator(
        retriever=FakeRetriever(),
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="Question cannot be empty or whitespace only"):
        orchestrator.answer("   ")

    with pytest.raises(ValueError, match="Question cannot be empty or whitespace only"):
        orchestrator.answer("")


def test_generation_error_propagation():
    chunk = create_dummy_chunk("Chunk")
    built_context = BuiltContext(formatted_text="Context", used_chunks=[chunk])

    retriever = FakeRetriever(chunks=[chunk])
    builder = FakeContextBuilder(context=built_context)
    generator = FakeAnswerGenerator()

    orchestrator = RAGOrchestrator(
        retriever=retriever, context_builder=builder, answer_generator=generator
    )

    with pytest.raises(GenerationError, match="Generation failed"):
        orchestrator.answer("trigger_generation_error")


def test_retrieval_error_propagation():
    orchestrator = RAGOrchestrator(
        retriever=FakeRetriever(),
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="Retrieval failed"):
        orchestrator.answer("trigger_retrieval_error")


def test_context_builder_error_propagation():
    chunk = create_dummy_chunk("trigger_context_error")
    retriever = FakeRetriever(chunks=[chunk])

    orchestrator = RAGOrchestrator(
        retriever=retriever,
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="Context building failed"):
        orchestrator.answer("Valid question")
