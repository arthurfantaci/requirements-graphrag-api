"""Tests for evaluation datasets module."""

from __future__ import annotations

from jama_mcp_server_graphrag.evaluation.datasets import (
    EvaluationSample,
    create_evaluation_dataset,
    get_sample_evaluation_data,
)


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""

    def test_sample_creation(self) -> None:
        """Test creating an evaluation sample."""
        sample = EvaluationSample(
            question="What is requirements traceability?",
            ground_truth="Traceability links requirements to artifacts.",
            contexts=["Context 1", "Context 2"],
            metadata={"topic": "traceability"},
        )

        assert sample.question == "What is requirements traceability?"
        assert sample.ground_truth == "Traceability links requirements to artifacts."
        assert len(sample.contexts) == 2
        assert sample.metadata["topic"] == "traceability"

    def test_sample_defaults(self) -> None:
        """Test sample with default values."""
        sample = EvaluationSample(
            question="Test question",
            ground_truth="Test answer",
        )

        assert sample.contexts == []
        assert sample.metadata == {}


class TestGetSampleEvaluationData:
    """Tests for get_sample_evaluation_data function."""

    def test_returns_samples(self) -> None:
        """Test that sample data is returned."""
        samples = get_sample_evaluation_data()

        assert len(samples) > 0
        assert all(isinstance(s, EvaluationSample) for s in samples)

    def test_samples_have_required_fields(self) -> None:
        """Test that all samples have required fields."""
        samples = get_sample_evaluation_data()

        for sample in samples:
            assert sample.question
            assert sample.ground_truth
            assert isinstance(sample.contexts, list)
            assert isinstance(sample.metadata, dict)

    def test_samples_have_metadata(self) -> None:
        """Test that samples have topic and difficulty metadata."""
        samples = get_sample_evaluation_data()

        for sample in samples:
            assert "topic" in sample.metadata
            assert "difficulty" in sample.metadata


class TestCreateEvaluationDataset:
    """Tests for create_evaluation_dataset function."""

    def test_create_default_dataset(self) -> None:
        """Test creating dataset with default samples."""
        dataset = create_evaluation_dataset()

        assert len(dataset) > 0
        assert all(isinstance(d, dict) for d in dataset)

    def test_dataset_format(self) -> None:
        """Test that dataset entries have correct format."""
        dataset = create_evaluation_dataset()

        for entry in dataset:
            assert "question" in entry
            assert "ground_truth" in entry
            assert "contexts" in entry
            assert "metadata" in entry

    def test_filter_by_topic(self) -> None:
        """Test filtering dataset by topic."""
        dataset = create_evaluation_dataset(filter_topic="traceability")

        assert len(dataset) > 0
        for entry in dataset:
            assert entry["metadata"]["topic"] == "traceability"

    def test_filter_by_difficulty(self) -> None:
        """Test filtering dataset by difficulty."""
        dataset = create_evaluation_dataset(filter_difficulty="basic")

        assert len(dataset) > 0
        for entry in dataset:
            assert entry["metadata"]["difficulty"] == "basic"

    def test_filter_no_matches(self) -> None:
        """Test filtering with no matches returns empty list."""
        dataset = create_evaluation_dataset(filter_topic="nonexistent")

        assert len(dataset) == 0

    def test_custom_samples(self) -> None:
        """Test creating dataset from custom samples."""
        custom_samples = [
            EvaluationSample(
                question="Custom question",
                ground_truth="Custom answer",
                metadata={"topic": "custom"},
            ),
        ]

        dataset = create_evaluation_dataset(custom_samples)

        assert len(dataset) == 1
        assert dataset[0]["question"] == "Custom question"
