"""Tests for observability comparison module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.observability_comparison import (
    PLATFORM_FEATURES,
    Platform,
    PlatformFeature,
    PlatformRecommendation,
    TrackingResult,
    UnifiedTracker,
    compare_platform_features,
    recommend_platform,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_langsmith_value(self):
        """Test LangSmith enum value."""
        assert Platform.LANGSMITH.value == "langsmith"

    def test_mlflow_value(self):
        """Test MLflow enum value."""
        assert Platform.MLFLOW.value == "mlflow"


class TestPlatformFeature:
    """Tests for PlatformFeature dataclass."""

    def test_create_feature(self):
        """Test creating a platform feature."""
        feature = PlatformFeature(
            name="Test Feature",
            langsmith_support=True,
            mlflow_support=False,
            langsmith_notes="Supported",
            mlflow_notes="Not supported",
        )
        assert feature.name == "Test Feature"
        assert feature.langsmith_support is True
        assert feature.mlflow_support is False
        assert feature.langsmith_notes == "Supported"
        assert feature.mlflow_notes == "Not supported"

    def test_default_notes(self):
        """Test default empty notes."""
        feature = PlatformFeature(
            name="Basic Feature",
            langsmith_support=True,
            mlflow_support=True,
        )
        assert feature.langsmith_notes == ""
        assert feature.mlflow_notes == ""


class TestTrackingResult:
    """Tests for TrackingResult dataclass."""

    def test_success_result(self):
        """Test successful tracking result."""
        result = TrackingResult(
            platform=Platform.LANGSMITH,
            success=True,
            run_id="run-123",
        )
        assert result.platform == Platform.LANGSMITH
        assert result.success is True
        assert result.run_id == "run-123"
        assert result.error is None

    def test_failure_result(self):
        """Test failed tracking result."""
        result = TrackingResult(
            platform=Platform.MLFLOW,
            success=False,
            error="Connection failed",
        )
        assert result.platform == Platform.MLFLOW
        assert result.success is False
        assert result.run_id is None
        assert result.error == "Connection failed"


class TestPlatformRecommendation:
    """Tests for PlatformRecommendation dataclass."""

    def test_create_recommendation(self):
        """Test creating a platform recommendation."""
        rec = PlatformRecommendation(
            recommended=Platform.LANGSMITH,
            confidence=0.85,
            reasons=["Native integration", "Easy setup"],
            considerations=["Cloud only"],
        )
        assert rec.recommended == Platform.LANGSMITH
        assert rec.confidence == 0.85
        assert len(rec.reasons) == 2
        assert len(rec.considerations) == 1

    def test_default_lists(self):
        """Test default empty lists."""
        rec = PlatformRecommendation(
            recommended=Platform.MLFLOW,
            confidence=0.70,
        )
        assert rec.reasons == []
        assert rec.considerations == []


class TestComparePlatformFeatures:
    """Tests for compare_platform_features function."""

    def test_returns_features_list(self):
        """Test that comparison returns features list."""
        comparison = compare_platform_features()
        assert "features" in comparison
        assert isinstance(comparison["features"], list)
        assert len(comparison["features"]) > 0

    def test_returns_summary(self):
        """Test that comparison returns summary."""
        comparison = compare_platform_features()
        assert "summary" in comparison
        assert "langsmith_advantages" in comparison["summary"]
        assert "mlflow_advantages" in comparison["summary"]
        assert "both_support" in comparison["summary"]

    def test_feature_structure(self):
        """Test structure of each feature in comparison."""
        comparison = compare_platform_features()
        for feature in comparison["features"]:
            assert "name" in feature
            assert "langsmith" in feature
            assert "mlflow" in feature
            assert "langsmith_notes" in feature
            assert "mlflow_notes" in feature

    def test_langsmith_advantages_populated(self):
        """Test that LangSmith advantages are populated."""
        comparison = compare_platform_features()
        # LangSmith should have advantages like trace visualization
        assert len(comparison["summary"]["langsmith_advantages"]) > 0

    def test_mlflow_advantages_populated(self):
        """Test that MLflow advantages are populated."""
        comparison = compare_platform_features()
        # MLflow should have advantages like self-hosting
        assert len(comparison["summary"]["mlflow_advantages"]) > 0

    def test_both_support_populated(self):
        """Test that shared features are populated."""
        comparison = compare_platform_features()
        # Both should support features like custom evaluators
        assert len(comparison["summary"]["both_support"]) > 0


class TestPlatformFeatures:
    """Tests for PLATFORM_FEATURES constant."""

    def test_features_not_empty(self):
        """Test that platform features list is not empty."""
        assert len(PLATFORM_FEATURES) > 0

    def test_all_features_are_platform_feature(self):
        """Test that all items are PlatformFeature instances."""
        for feature in PLATFORM_FEATURES:
            assert isinstance(feature, PlatformFeature)

    def test_has_self_hosting_feature(self):
        """Test that self-hosting feature exists."""
        feature_names = [f.name for f in PLATFORM_FEATURES]
        assert "Self-hosted option" in feature_names

    def test_has_trace_visualization_feature(self):
        """Test that trace visualization feature exists."""
        feature_names = [f.name for f in PLATFORM_FEATURES]
        assert "Trace visualization" in feature_names


class TestRecommendPlatform:
    """Tests for recommend_platform function."""

    def test_default_recommends_langsmith(self):
        """Test default parameters recommend LangSmith."""
        rec = recommend_platform()
        assert rec.recommended == Platform.LANGSMITH
        assert rec.confidence > 0

    def test_self_hosting_recommends_mlflow(self):
        """Test self-hosting requirement recommends MLflow."""
        rec = recommend_platform(needs_self_hosting=True)
        assert rec.recommended == Platform.MLFLOW
        assert "self-hosting" in rec.reasons[0].lower()

    def test_budget_sensitive_adds_consideration(self):
        """Test budget sensitive adds MLflow considerations."""
        rec = recommend_platform(budget_sensitive=True)
        assert len(rec.considerations) > 0
        consideration_text = " ".join(rec.considerations).lower()
        assert "free" in consideration_text or "open-source" in consideration_text

    def test_trace_visualization_adds_langsmith_score(self):
        """Test trace visualization adds to LangSmith score."""
        rec_with = recommend_platform(needs_trace_visualization=True)
        rec_without = recommend_platform(needs_trace_visualization=False)

        # Both should have reasons but different content
        with_reasons = " ".join(rec_with.reasons).lower()
        without_reasons = " ".join(rec_without.reasons).lower()

        if rec_with.recommended == Platform.LANGSMITH:
            assert "trace" in with_reasons or "visualization" in with_reasons

    def test_prompt_hub_adds_langsmith_score(self):
        """Test prompt hub requirement adds to LangSmith score."""
        rec = recommend_platform(needs_prompt_hub=True)
        reasons_text = " ".join(rec.reasons).lower()
        assert "langchainhub" in reasons_text or "prompt" in reasons_text

    def test_cost_tracking_adds_langsmith_score(self):
        """Test cost tracking adds to LangSmith score."""
        rec = recommend_platform(needs_cost_tracking=True)
        reasons_text = " ".join(rec.reasons).lower()
        assert "cost" in reasons_text

    def test_langchain_native_adds_langsmith_score(self):
        """Test LangChain native adds to LangSmith score."""
        rec = recommend_platform(langchain_native=True)
        reasons_text = " ".join(rec.reasons).lower()
        assert "langchain" in reasons_text

    def test_confidence_capped_at_95(self):
        """Test that confidence is capped at 95%."""
        rec = recommend_platform(
            needs_trace_visualization=True,
            needs_prompt_hub=True,
            needs_cost_tracking=True,
            langchain_native=True,
        )
        assert rec.confidence <= 0.95

    def test_recommendation_has_reasons(self):
        """Test that recommendation always has reasons."""
        rec = recommend_platform()
        assert len(rec.reasons) > 0


class TestUnifiedTracker:
    """Tests for UnifiedTracker class."""

    def test_init_default_platforms(self):
        """Test default initialization includes both platforms."""
        tracker = UnifiedTracker()
        assert Platform.LANGSMITH in tracker.platforms
        assert Platform.MLFLOW in tracker.platforms

    def test_init_custom_platforms(self):
        """Test custom platform selection."""
        tracker = UnifiedTracker(platforms=[Platform.LANGSMITH])
        assert tracker.platforms == [Platform.LANGSMITH]

    def test_init_run_name_auto_generated(self):
        """Test that run name is auto-generated if not provided."""
        tracker = UnifiedTracker()
        assert tracker.run_name.startswith("unified-run-")

    def test_init_custom_run_name(self):
        """Test custom run name."""
        tracker = UnifiedTracker(run_name="my-test-run")
        assert tracker.run_name == "my-test-run"

    def test_init_experiment_name(self):
        """Test experiment name configuration."""
        tracker = UnifiedTracker(experiment_name="custom-experiment")
        assert tracker.experiment_name == "custom-experiment"

    def test_context_manager_sync(self):
        """Test synchronous context manager."""
        with patch.object(UnifiedTracker, "_start_langsmith"):
            with patch.object(UnifiedTracker, "_start_mlflow"):
                tracker = UnifiedTracker(platforms=[])
                with tracker as t:
                    assert t is tracker
                    assert t._start_time is not None

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test asynchronous context manager."""
        with patch.object(UnifiedTracker, "_start_langsmith"):
            with patch.object(UnifiedTracker, "_start_mlflow"):
                tracker = UnifiedTracker(platforms=[])
                async with tracker as t:
                    assert t is tracker

    def test_get_results_empty_initially(self):
        """Test that results are empty before tracking."""
        tracker = UnifiedTracker(platforms=[])
        assert tracker.get_results() == []

    def test_log_params_calls_mlflow(self):
        """Test that log_params calls MLflow tracker."""
        mock_mlflow_tracker = MagicMock()
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = mock_mlflow_tracker

        tracker.log_params({"key": "value"})

        mock_mlflow_tracker.log_params.assert_called_once_with({"key": "value"})

    def test_log_params_handles_no_mlflow(self):
        """Test log_params when MLflow tracker is None."""
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = None
        # Should not raise
        tracker.log_params({"key": "value"})

    def test_log_metrics_calls_mlflow(self):
        """Test that log_metrics calls MLflow tracker."""
        mock_mlflow_tracker = MagicMock()
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = mock_mlflow_tracker

        tracker.log_metrics({"accuracy": 0.9}, step=1)

        mock_mlflow_tracker.log_metrics.assert_called_once_with({"accuracy": 0.9}, step=1)

    def test_log_metric_single(self):
        """Test logging a single metric."""
        mock_mlflow_tracker = MagicMock()
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = mock_mlflow_tracker

        tracker.log_metric("accuracy", 0.95, step=2)

        mock_mlflow_tracker.log_metrics.assert_called_once_with({"accuracy": 0.95}, step=2)

    def test_log_artifact_calls_mlflow(self):
        """Test that log_artifact calls MLflow tracker."""
        mock_mlflow_tracker = MagicMock()
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = mock_mlflow_tracker

        tracker.log_artifact("/path/to/file", "artifacts")

        mock_mlflow_tracker.log_artifact.assert_called_once_with("/path/to/file", "artifacts")

    def test_log_artifact_handles_no_mlflow(self):
        """Test log_artifact when MLflow tracker is None."""
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = None
        # Should not raise
        tracker.log_artifact("/path/to/file")

    def test_start_langsmith_success(self):
        """Test successful LangSmith initialization."""
        with patch("langsmith.Client") as mock_client:
            mock_client.return_value = MagicMock()
            tracker = UnifiedTracker(run_name="test-run")
            tracker._results = []

            tracker._start_langsmith()

            assert len(tracker._results) == 1
            assert tracker._results[0].platform == Platform.LANGSMITH
            assert tracker._results[0].success is True

    def test_start_langsmith_failure(self):
        """Test LangSmith initialization failure."""
        with patch(
            "langsmith.Client",
            side_effect=Exception("API key invalid"),
        ):
            tracker = UnifiedTracker()
            tracker._results = []

            tracker._start_langsmith()

            assert len(tracker._results) == 1
            assert tracker._results[0].platform == Platform.LANGSMITH
            assert tracker._results[0].success is False
            assert "API key invalid" in tracker._results[0].error

    def test_start_mlflow_when_unavailable(self):
        """Test MLflow initialization when not available."""
        # Create a mock module that reports MLFLOW_AVAILABLE as False
        mock_mlflow_module = MagicMock()
        mock_mlflow_module.MLFLOW_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {"jama_mcp_server_graphrag.mlflow_tracking": mock_mlflow_module},
        ):
            tracker = UnifiedTracker()
            tracker._results = []

            tracker._start_mlflow()

            # Should have a result indicating MLflow not available
            assert len(tracker._results) == 1
            assert tracker._results[0].platform == Platform.MLFLOW
            assert tracker._results[0].success is False

    def test_exit_logs_duration(self):
        """Test that exit logs duration metric."""
        mock_mlflow_tracker = MagicMock()
        tracker = UnifiedTracker(platforms=[])
        tracker._mlflow_tracker = mock_mlflow_tracker
        tracker._start_time = 100.0

        with patch("time.time", return_value=105.0):
            tracker.__exit__(None, None, None)

        # Check duration was logged
        mock_mlflow_tracker.log_metrics.assert_called()
