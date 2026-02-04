#!/usr/bin/env python3
"""Create the agentic evaluation dataset in LangSmith.

This script creates a dataset for evaluating the agentic RAG system.
It includes examples for:
- Multi-hop reasoning questions
- Tool selection scenarios
- Iteration efficiency tests

Usage:
    python scripts/create_agentic_dataset.py

Requires:
    LANGSMITH_API_KEY environment variable
"""

from __future__ import annotations

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def create_agentic_dataset() -> None:
    """Create the agentic evaluation dataset."""
    from langsmith import Client

    client = Client()

    dataset_name = "graphrag-agentic-eval"
    dataset_description = """Evaluation dataset for the GraphRAG Agentic RAG system.

Contains examples testing:
- Multi-hop reasoning: Questions requiring information synthesis across sources
- Tool selection: Scenarios with expected tool usage patterns
- Iteration efficiency: Questions with known complexity for iteration testing
- Critic calibration: Examples with expert quality assessments

Use with agentic_evaluators.py for comprehensive agent evaluation.
"""

    # Check if dataset already exists
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        print(f"Dataset '{dataset_name}' already exists. Skipping creation.")
        print(f"Dataset ID: {existing[0].id}")
        return

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=dataset_description,
    )
    print(f"Created dataset: {dataset_name} (ID: {dataset.id})")

    # Define evaluation examples
    examples = [
        # =================================================================
        # MULTI-HOP REASONING EXAMPLES
        # =================================================================
        {
            "inputs": {
                "question": "How does requirements traceability help with impact analysis when a safety standard changes?",
                "complexity": "multi_hop",
            },
            "outputs": {
                "expected_answer": "Requirements traceability creates bidirectional links between requirements and downstream artifacts (design, code, tests). When a safety standard changes, you can trace from the affected requirements to all derived work items, allowing you to identify exactly which designs, code modules, and tests need updates.",
                "expected_tools": ["graph_search", "explore_entity"],
                "expected_iterations": 3,
                "required_reasoning_steps": [
                    "Understand what requirements traceability provides (bidirectional links)",
                    "Connect traceability to impact analysis capability",
                    "Link safety standards to requirements",
                    "Explain how changes propagate through the trace chain",
                ],
            },
            "metadata": {"category": "multi_hop", "difficulty": "hard"},
        },
        {
            "inputs": {
                "question": "What is the relationship between verification and validation in the context of the V-model?",
                "complexity": "multi_hop",
            },
            "outputs": {
                "expected_answer": "In the V-model, verification activities (reviews, inspections) occur on the left side during development phases, ensuring work products meet specifications. Validation activities (testing) occur on the right side, confirming the final product meets user needs. The horizontal connections show how each development phase has a corresponding test phase.",
                "expected_tools": ["graph_search"],
                "expected_iterations": 2,
                "required_reasoning_steps": [
                    "Define verification in V-model context",
                    "Define validation in V-model context",
                    "Explain the V-model structure (left vs right side)",
                    "Connect development phases to test phases",
                ],
            },
            "metadata": {"category": "multi_hop", "difficulty": "medium"},
        },
        {
            "inputs": {
                "question": "How do user stories relate to acceptance criteria in agile requirements management?",
                "complexity": "multi_hop",
            },
            "outputs": {
                "expected_answer": "User stories define what a user wants to accomplish ('As a... I want... So that...'). Acceptance criteria define the specific conditions that must be met for the story to be considered complete. Together, they form a testable requirement: the user story provides context and business value, while acceptance criteria provide measurable success conditions.",
                "expected_tools": ["graph_search", "search_definitions"],
                "expected_iterations": 2,
                "required_reasoning_steps": [
                    "Define user story format and purpose",
                    "Define acceptance criteria purpose",
                    "Connect user stories to acceptance criteria",
                    "Explain how they work together for testability",
                ],
            },
            "metadata": {"category": "multi_hop", "difficulty": "medium"},
        },
        # =================================================================
        # TOOL SELECTION EXAMPLES
        # =================================================================
        {
            "inputs": {
                "question": "What is requirements traceability?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Requirements traceability is the ability to describe and follow the life of a requirement in both forwards and backwards direction through the development lifecycle.",
                "expected_tools": ["graph_search"],
                "expected_iterations": 1,
            },
            "metadata": {"category": "tool_selection", "difficulty": "easy"},
        },
        {
            "inputs": {
                "question": "List all webinars about traceability",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Query executed to list webinars with traceability content.",
                "expected_tools": ["text2cypher"],
                "expected_iterations": 1,
            },
            "metadata": {"category": "tool_selection", "difficulty": "easy"},
        },
        {
            "inputs": {
                "question": "Tell me about DO-178C and its relationship to requirements management",
                "complexity": "complex",
            },
            "outputs": {
                "expected_answer": "DO-178C is an aviation software safety standard that requires rigorous requirements management including bidirectional traceability, derived requirements tracking, and formal verification of safety-critical requirements.",
                "expected_tools": ["graph_search", "explore_entity", "lookup_standard"],
                "expected_iterations": 3,
            },
            "metadata": {"category": "tool_selection", "difficulty": "hard"},
        },
        {
            "inputs": {
                "question": "What does MBSE stand for?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "MBSE stands for Model-Based Systems Engineering.",
                "expected_tools": ["search_definitions"],
                "expected_iterations": 1,
            },
            "metadata": {"category": "tool_selection", "difficulty": "easy"},
        },
        # =================================================================
        # ITERATION EFFICIENCY EXAMPLES
        # =================================================================
        {
            "inputs": {
                "question": "What are the key benefits of requirements management?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Key benefits include improved communication, reduced rework, better traceability, regulatory compliance, and faster time to market.",
                "expected_tools": ["graph_search"],
                "expected_iterations": 1,
                "expert_quality": 0.9,
            },
            "metadata": {"category": "iteration_efficiency", "difficulty": "easy"},
        },
        {
            "inputs": {
                "question": "How should an organization implement requirements traceability for a medical device project?",
                "complexity": "complex",
            },
            "outputs": {
                "expected_answer": "For medical devices, implement traceability per FDA 21 CFR Part 820 and IEC 62304. This includes: 1) Establishing a traceability matrix linking user needs to design inputs, outputs, and verification. 2) Using a requirements management tool that supports trace links. 3) Maintaining bidirectional traceability from user needs through risk analysis, design, verification, and validation. 4) Creating Design History File documentation.",
                "expected_tools": ["graph_search", "explore_entity", "lookup_standard"],
                "expected_iterations": 3,
                "expert_quality": 0.85,
            },
            "metadata": {"category": "iteration_efficiency", "difficulty": "hard"},
        },
        # =================================================================
        # CRITIC CALIBRATION EXAMPLES
        # =================================================================
        {
            "inputs": {
                "question": "What is a design input vs design output?",
                "complexity": "medium",
            },
            "outputs": {
                "expected_answer": "Design inputs are the requirements and specifications that guide design work (what needs to be built). Design outputs are the results of design activities (how it will be built) - drawings, specifications, procedures that describe the design and meet the design inputs.",
                "expected_tools": ["graph_search", "search_definitions"],
                "expected_iterations": 2,
                "expert_quality": 0.9,
                "expert_missing_aspects": [],
            },
            "metadata": {"category": "critic_calibration", "difficulty": "medium"},
        },
        {
            "inputs": {
                "question": "Explain the relationship between hazard analysis and safety requirements",
                "complexity": "complex",
            },
            "outputs": {
                "expected_answer": "Hazard analysis identifies potential hazards and their causes. This feeds into safety requirements that mitigate or eliminate hazards. The relationship is bidirectional: hazard analysis drives safety requirement creation, and safety requirements must trace back to specific hazards to ensure complete coverage.",
                "expected_tools": ["graph_search", "explore_entity"],
                "expected_iterations": 3,
                "expert_quality": 0.75,
                "expert_missing_aspects": [
                    "Risk assessment methodology (FMEA, FTA)",
                    "Residual risk acceptance criteria",
                ],
            },
            "metadata": {"category": "critic_calibration", "difficulty": "hard"},
        },
        # =================================================================
        # ADDITIONAL VARIED EXAMPLES
        # =================================================================
        {
            "inputs": {
                "question": "How do you measure requirements quality?",
                "complexity": "medium",
            },
            "outputs": {
                "expected_answer": "Requirements quality is measured through attributes like: completeness, consistency, correctness, unambiguity, verifiability, traceability, and feasibility. Metrics include defect density, requirements volatility, and review pass rates.",
                "expected_tools": ["graph_search"],
                "expected_iterations": 2,
                "expert_quality": 0.85,
            },
            "metadata": {"category": "general", "difficulty": "medium"},
        },
        {
            "inputs": {
                "question": "What is the difference between functional and non-functional requirements?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Functional requirements describe what the system should do (features, behaviors, operations). Non-functional requirements describe how the system should perform (performance, security, usability, reliability constraints).",
                "expected_tools": ["graph_search", "search_definitions"],
                "expected_iterations": 1,
                "expert_quality": 0.95,
            },
            "metadata": {"category": "general", "difficulty": "easy"},
        },
        {
            "inputs": {
                "question": "Explain requirements decomposition and allocation in systems engineering",
                "complexity": "complex",
            },
            "outputs": {
                "expected_answer": "Requirements decomposition breaks high-level requirements into more detailed sub-requirements. Allocation assigns these decomposed requirements to specific system components, subsystems, or elements. This creates a hierarchical trace from stakeholder needs through system requirements to component specifications.",
                "expected_tools": ["graph_search", "explore_entity"],
                "expected_iterations": 3,
                "expert_quality": 0.8,
                "required_reasoning_steps": [
                    "Define decomposition process",
                    "Define allocation process",
                    "Connect decomposition to allocation",
                    "Explain hierarchical traceability",
                ],
            },
            "metadata": {"category": "multi_hop", "difficulty": "hard"},
        },
        {
            "inputs": {
                "question": "What tools does Jama Software offer for requirements management?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Jama Connect is a requirements management platform offering: requirements authoring and management, traceability matrices, impact analysis, review and approval workflows, test management integration, and compliance documentation.",
                "expected_tools": ["graph_search"],
                "expected_iterations": 1,
                "expert_quality": 0.9,
            },
            "metadata": {"category": "tool_selection", "difficulty": "easy"},
        },
        {
            "inputs": {
                "question": "How many articles discuss ISO 26262?",
                "complexity": "simple",
            },
            "outputs": {
                "expected_answer": "Query executed to count articles mentioning ISO 26262.",
                "expected_tools": ["text2cypher"],
                "expected_iterations": 1,
            },
            "metadata": {"category": "tool_selection", "difficulty": "easy"},
        },
    ]

    # Create examples in bulk
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    print(f"Created {len(examples)} examples in dataset '{dataset_name}'")
    print("\nExample categories:")
    categories = {}
    for ex in examples:
        cat = ex.get("metadata", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")


if __name__ == "__main__":
    create_agentic_dataset()
