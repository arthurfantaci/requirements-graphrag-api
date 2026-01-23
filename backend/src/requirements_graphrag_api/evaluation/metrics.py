"""RAG evaluation metrics and prompts.

Provides LLM-as-judge prompts for evaluating RAG pipeline quality:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved contexts relevant to the question?
- Context Recall: Does the context contain the ground truth?

These prompts follow the RAGAS evaluation framework patterns.
"""

from __future__ import annotations

# =============================================================================
# RAG EVALUATION PROMPTS
# =============================================================================

FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an answer to its source context.

Given the following:
- Context: {context}
- Question: {question}
- Answer: {answer}

Evaluate whether the answer is fully supported by the context provided.
An answer is faithful if every claim made can be directly verified from the context.

Score from 0.0 to 1.0:
- 1.0: All claims are directly supported by the context
- 0.5: Some claims are supported, others are not verifiable
- 0.0: Claims contradict or are not present in the context

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

ANSWER_RELEVANCY_PROMPT = """You are evaluating the relevancy of an answer to a question.

Given the following:
- Question: {question}
- Answer: {answer}

Evaluate whether the answer directly addresses the question asked.
An answer is relevant if it provides information that helps answer the question.

Score from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer partially addresses the question or includes irrelevant info
- 0.0: Answer does not address the question at all

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of retrieved contexts for a question.

Given the following:
- Question: {question}
- Retrieved Contexts: {contexts}

Evaluate what proportion of the retrieved contexts are relevant to answering the question.
A context is relevant if it contains information useful for answering the question.

Score from 0.0 to 1.0:
- 1.0: All retrieved contexts are relevant
- 0.5: About half of the contexts are relevant
- 0.0: None of the contexts are relevant

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

CONTEXT_RECALL_PROMPT = """You are evaluating context recall against ground truth.

Given the following:
- Question: {question}
- Retrieved Contexts: {contexts}
- Ground Truth Answer: {ground_truth}

Evaluate whether the retrieved contexts contain the information needed to generate the ground truth.
High recall means the contexts cover all key facts from the ground truth.

Score from 0.0 to 1.0:
- 1.0: Contexts contain all information needed for ground truth
- 0.5: Contexts contain some but not all needed information
- 0.0: Contexts do not contain information from ground truth

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""
