from dataclasses import dataclass
from typing import Collection, List, Optional

from core.answer_generation import AnswerGenerator
from core.context import ContextBuilder
from core.retrieval import RetrievedChunk, Retriever

NO_EVIDENCE_RESPONSE = "I cannot answer this question because no relevant information was found in the documents."


@dataclass
class OrchestrationResult:
    answer_text: str
    evidence: List[RetrievedChunk]


class RAGOrchestrator:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.answer_generator = answer_generator

    def answer(
        self,
        question: str,
        document_ids: Optional[Collection[str]] = None,
    ) -> OrchestrationResult:
        """
        Coordinates retrieval, context building, and answer generation.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace only.")

        clean_question = question.strip()

        # 1. Retrieve raw chunks
        retrieved_chunks = self.retriever.retrieve(
            clean_question, document_ids=document_ids
        )

        # 2. Build constrained context
        built_context = self.context_builder.build(retrieved_chunks)

        # 3. Check evidence
        if not built_context.used_chunks:
            return OrchestrationResult(
                answer_text=NO_EVIDENCE_RESPONSE,
                evidence=[],
            )

        # 4. Generate answer
        generated_answer = self.answer_generator.generate(
            question=clean_question,
            context=built_context,
        )

        # 5. Return final result preserving ONLY the used chunks
        return OrchestrationResult(
            answer_text=generated_answer.answer_text,
            evidence=built_context.used_chunks,
        )
