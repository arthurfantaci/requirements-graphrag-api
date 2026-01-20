"""Tests for MLflow tracking integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jama_mcp_server_graphrag.mlflow_tracking import (
    MLFLOW_AVAILABLE,
    MLflowConfig,
    MLflowTracker,
    compare_runs,
    configure_mlflow,
    get_best_run,
    get_mlflow_status,
    log_rag_evaluation,
)


class TestMLflowConfig:
    """Tests for MLflowConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MLflowConfig()
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "jama-graphrag-evaluation"
        assert config.artifact_location is None
        assert config.registry_uri is None
        assert config.tags == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MLflowConfig(
            tracking_uri="http://custom:8080",
            experiment_name="custom-experiment",
            artifact_location="/artifacts",
            registry_uri="http://registry:5000",
            tags={"env": "test"},
        )
        assert config.tracking_uri == "http://custom:8080"
        assert config.experiment_name == "custom-experiment"
        assert config.artifact_location == "/artifacts"
        assert config.registry_uri == "http://registry:5000"
        assert config.tags == {"env": "test"}

    def test_immutable(self):
        """Test that config is immutable (frozen)."""
        config = MLflowConfig()
        with pytest.raises(AttributeError):
            config.tracking_uri = "new-uri"


class TestConfigureMlflow:
    """Tests for configure_mlflow function."""

    def test_returns_false_when_mlflow_unavailable(self):
        """Test that configure_mlflow returns False when MLflow is not installed."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            result = configure_mlflow()
            assert result is False

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_creates_new_experiment(self, mock_mlflow):
        """Test that a new experiment is created when it doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        result = configure_mlflow(
            tracking_uri="http://test:5000",
            experiment_name="test-experiment",
        )

        assert result is True
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://test:5000")
        mock_mlflow.create_experiment.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_uses_existing_experiment(self, mock_mlflow):
        """Test that existing experiment is used when it exists."""
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()

        result = configure_mlflow(experiment_name="existing-experiment")

        assert result is True
        mock_mlflow.create_experiment.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with("existing-experiment")

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_handles_exception(self, mock_mlflow):
        """Test that exceptions are handled gracefully."""
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connection failed")

        result = configure_mlflow()

        assert result is False


class TestGetMlflowStatus:
    """Tests for get_mlflow_status function."""

    def test_returns_unavailable_when_mlflow_not_installed(self):
        """Test status when MLflow is not available."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            status = get_mlflow_status()

            assert status["available"] is False
            assert status["tracking_uri"] is None
            assert status["experiment"] is None
            assert "error" in status

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_returns_status_without_active_run(self, mock_mlflow):
        """Test status when no active run."""
        mock_mlflow.get_tracking_uri.return_value = "http://localhost:5000"
        mock_mlflow.active_run.return_value = None

        status = get_mlflow_status()

        assert status["available"] is True
        assert status["tracking_uri"] == "http://localhost:5000"
        assert status["experiment"] is None
        assert status["active_run"] is None

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_returns_status_with_active_run(self, mock_mlflow):
        """Test status with active run."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_run.info.experiment_id = "exp-123"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "http://localhost:5000"

        mock_experiment = MagicMock()
        mock_experiment.name = "test-experiment"
        mock_mlflow.get_experiment.return_value = mock_experiment

        status = get_mlflow_status()

        assert status["available"] is True
        assert status["active_run"] == "test-run-id"


class TestMLflowTracker:
    """Tests for MLflowTracker class."""

    def test_init_default_values(self):
        """Test default initialization."""
        tracker = MLflowTracker()
        assert tracker.run_name is None
        assert tracker.experiment_name is None
        assert tracker.tags == {}
        assert tracker.nested is False

    def test_init_custom_values(self):
        """Test custom initialization."""
        tracker = MLflowTracker(
            run_name="test-run",
            experiment_name="test-experiment",
            tags={"env": "test"},
            nested=True,
        )
        assert tracker.run_name == "test-run"
        assert tracker.experiment_name == "test-experiment"
        assert tracker.tags == {"env": "test"}
        assert tracker.nested is True

    def test_context_manager_when_mlflow_unavailable(self):
        """Test context manager when MLflow is not available."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            tracker = MLflowTracker(run_name="test")
            with tracker as t:
                assert t is tracker
                assert t.run_id is None

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_context_manager_starts_run(self, mock_mlflow):
        """Test that context manager starts MLflow run."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker(run_name="test-run", experiment_name="test-exp")
        with tracker:
            pass

        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_log_params(self, mock_mlflow):
        """Test logging parameters."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.log_params({"model": "gpt-4", "k": 6})

        mock_mlflow.log_params.assert_called_once_with({"model": "gpt-4", "k": "6"})

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.90}, step=1)

        mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.95, "f1": 0.90}, step=1)

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_log_metric(self, mock_mlflow):
        """Test logging single metric."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.log_metric("accuracy", 0.95, step=2)

        # Check the metric was logged (duration is also logged on exit)
        mock_mlflow.log_metric.assert_any_call("accuracy", 0.95, step=2)

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_log_artifact(self, mock_mlflow):
        """Test logging artifact."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.log_artifact("/path/to/file.json", "artifacts")

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/file.json", "artifacts")

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_log_dict(self, mock_mlflow):
        """Test logging dictionary."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.log_dict({"key": "value"}, "config.json")

        mock_mlflow.log_dict.assert_called_once_with({"key": "value"}, "config.json")

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_set_tag(self, mock_mlflow):
        """Test setting tag."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            tracker.set_tag("env", "production")

        mock_mlflow.set_tag.assert_called_once_with("env", "production")

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_run_id_property(self, mock_mlflow):
        """Test run_id property."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-456"
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        with tracker:
            assert tracker.run_id == "test-run-456"

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_logs_error_on_exception(self, mock_mlflow):
        """Test that errors are logged on exception."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker()
        try:
            with tracker:
                raise ValueError("Test error")
        except ValueError:
            pass

        mock_mlflow.set_tag.assert_called_once_with("error", "Test error")


class TestLogRagEvaluation:
    """Tests for log_rag_evaluation function."""

    def test_returns_none_when_mlflow_unavailable(self):
        """Test returns None when MLflow not available."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            result = log_rag_evaluation(
                question="What is X?",
                answer="X is Y.",
                contexts=["Context 1"],
                metrics={"accuracy": 0.9},
            )
            assert result is None

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_logs_evaluation_data(self, mock_mlflow):
        """Test that evaluation data is logged correctly."""
        mock_run = MagicMock()
        mock_run.info.run_id = "eval-run-123"
        mock_mlflow.start_run.return_value = mock_run

        result = log_rag_evaluation(
            question="What is requirements traceability?",
            answer="Requirements traceability is...",
            contexts=["Context about traceability"],
            metrics={"faithfulness": 0.85, "relevancy": 0.90},
            ground_truth="Expected answer",
            latency_ms=150.5,
            run_name="test-eval",
        )

        assert result == "eval-run-123"
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called()


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_returns_empty_when_mlflow_unavailable(self):
        """Test returns empty dict when MLflow not available."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            result = compare_runs(["run-1", "run-2"])
            assert result == {}

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.MlflowClient")
    def test_compares_runs(self, mock_client_class):
        """Test comparing multiple runs."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_run1 = MagicMock()
        mock_run1.data.metrics = {"accuracy": 0.9, "f1": 0.85}
        mock_run2 = MagicMock()
        mock_run2.data.metrics = {"accuracy": 0.92, "f1": 0.88}

        mock_client.get_run.side_effect = [mock_run1, mock_run2]

        result = compare_runs(["run-1", "run-2"])

        assert "run-1" in result
        assert "run-2" in result
        assert result["run-1"]["accuracy"] == 0.9
        assert result["run-2"]["accuracy"] == 0.92

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.MlflowClient")
    def test_filters_metrics(self, mock_client_class):
        """Test filtering specific metrics."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_run = MagicMock()
        mock_run.data.metrics = {"accuracy": 0.9, "f1": 0.85, "loss": 0.1}

        mock_client.get_run.return_value = mock_run

        result = compare_runs(["run-1"], metrics=["accuracy"])

        assert "accuracy" in result["run-1"]
        assert "f1" not in result["run-1"]
        assert "loss" not in result["run-1"]


class TestGetBestRun:
    """Tests for get_best_run function."""

    def test_returns_none_when_mlflow_unavailable(self):
        """Test returns None when MLflow not available."""
        with patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", False):
            result = get_best_run("experiment", "accuracy")
            assert result is None

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_returns_none_when_experiment_not_found(self, mock_mlflow):
        """Test returns None when experiment doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        result = get_best_run("nonexistent", "accuracy")

        assert result is None

    @patch("jama_mcp_server_graphrag.mlflow_tracking.MLFLOW_AVAILABLE", True)
    @patch("jama_mcp_server_graphrag.mlflow_tracking.mlflow")
    def test_returns_best_run(self, mock_mlflow):
        """Test returning best run."""
        import pandas as pd

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_runs = pd.DataFrame(
            [
                {
                    "run_id": "best-run",
                    "metrics.accuracy": 0.95,
                    "params.model": "gpt-4",
                }
            ]
        )
        mock_mlflow.search_runs.return_value = mock_runs

        result = get_best_run("test-experiment", "accuracy", maximize=True)

        assert result is not None
        assert result["run_id"] == "best-run"
        assert result["metric_value"] == 0.95
        mock_mlflow.search_runs.assert_called_once()


class TestMlflowAvailable:
    """Tests for MLFLOW_AVAILABLE constant."""

    def test_mlflow_available_is_boolean(self):
        """Test that MLFLOW_AVAILABLE is a boolean."""
        assert isinstance(MLFLOW_AVAILABLE, bool)
