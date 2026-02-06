"""Tests for conversational intent handler."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.core.conversation import (
    _EMPTY_HISTORY_RESPONSE,
    _format_conversation_history,
    handle_conversational,
    stream_conversational_events,
)
from requirements_graphrag_api.core.generation import StreamEventType
from requirements_graphrag_api.prompts import PromptName
from requirements_graphrag_api.prompts.definitions import PROMPT_DEFINITIONS

if TYPE_CHECKING:
    from requirements_graphrag_api.config import AppConfig


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_history() -> list[dict[str, str]]:
    """Standard multi-turn conversation history."""
    return [
        {"role": "user", "content": "What is traceability?"},
        {"role": "assistant", "content": "Traceability is the ability to track requirements."},
        {"role": "user", "content": "How is it implemented?"},
        {"role": "assistant", "content": "It is implemented through a traceability matrix."},
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock ChatOpenAI compatible with LangChain's pipe operator.

    For ``handle_conversational``: uses ``prompt_template | llm`` (RunnableSequence).
    LangChain calls ``ainvoke`` on the mock and reads ``.content`` from the result.

    For ``stream_conversational_events``: uses ``llm.astream(messages)`` directly.
    """
    from langchain_core.messages import AIMessage as _AIMessage

    llm = MagicMock()
    _answer = "Your first question was about traceability."

    # For handle_conversational (uses chain.ainvoke via prompt | llm)
    llm.ainvoke = AsyncMock(return_value=_AIMessage(content=_answer))

    # For stream_conversational_events (uses llm.astream)
    async def mock_astream(*_args: Any, **_kwargs: Any):
        tokens = ["Your ", "first ", "question ", "was ", "about ", "traceability."]
        for token in tokens:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    llm.astream = mock_astream

    return llm


# ── _format_conversation_history ─────────────────────────────────────────────


class TestFormatConversationHistory:
    """Tests for conversation history formatting."""

    def test_empty_history(self) -> None:
        result = _format_conversation_history([])
        assert result == "(No conversation history available)"

    def test_single_turn(self) -> None:
        history = [{"role": "user", "content": "Hello"}]
        result = _format_conversation_history(history)
        assert result == "1. User: Hello"

    def test_multi_turn(self, sample_history: list[dict[str, str]]) -> None:
        result = _format_conversation_history(sample_history)
        lines = result.split("\n")
        assert len(lines) == 4
        assert lines[0] == "1. User: What is traceability?"
        assert lines[1] == "2. Assistant: Traceability is the ability to track requirements."
        assert lines[2] == "3. User: How is it implemented?"
        assert lines[3] == "4. Assistant: It is implemented through a traceability matrix."

    def test_long_history(self) -> None:
        """Test formatting with 40+ messages."""
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(40)
        ]
        result = _format_conversation_history(history)
        lines = result.split("\n")
        assert len(lines) == 40
        assert lines[0].startswith("1. User:")
        assert lines[39].startswith("40. Assistant:")

    def test_missing_content_key(self) -> None:
        """Test handling of messages without content key."""
        history = [{"role": "user"}]
        result = _format_conversation_history(history)
        assert result == "1. User: "

    def test_missing_role_key(self) -> None:
        """Test handling of messages without role key defaults to Assistant."""
        history = [{"content": "Hello"}]
        result = _format_conversation_history(history)
        assert result == "1. Assistant: Hello"


# ── handle_conversational ────────────────────────────────────────────────────


class TestHandleConversational:
    """Tests for handle_conversational function.

    ``handle_conversational`` uses ``prompt_template | llm`` (LangChain pipe).
    MagicMock doesn't implement the Runnable protocol, so we mock the chain
    at the ``get_prompt_sync`` level — returning a mock prompt whose ``__or__``
    yields a mock chain with a controlled ``ainvoke``.
    """

    @staticmethod
    def _patch_chain(answer: str = "Your first question was about traceability."):
        """Create patches that make the prompt|llm chain return ``answer``."""
        mock_result = MagicMock()
        mock_result.content = answer

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_result)

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        return (
            patch(
                "requirements_graphrag_api.core.conversation.get_prompt_sync",
                return_value=mock_prompt,
            ),
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI"),
        )

    @pytest.mark.asyncio
    async def test_returns_answer_and_run_id(
        self, mock_config: AppConfig, sample_history: list[dict[str, str]]
    ) -> None:
        """Test that handler returns (answer, run_id) tuple."""
        mock_run_tree = MagicMock()
        mock_run_tree.id = "test-run-id-123"

        prompt_patch, llm_patch = self._patch_chain()
        with (
            prompt_patch,
            llm_patch,
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=mock_run_tree,
            ),
        ):
            answer, run_id = await handle_conversational(
                mock_config, "What was my first question?", sample_history
            )

        assert answer == "Your first question was about traceability."
        assert run_id == "test-run-id-123"

    @pytest.mark.asyncio
    async def test_empty_history_returns_graceful_message(self, mock_config: AppConfig) -> None:
        """Test empty history returns predefined message without LLM call."""
        with patch(
            "requirements_graphrag_api.core.conversation.get_current_run_tree",
            return_value=None,
        ):
            answer, run_id = await handle_conversational(mock_config, "What did we discuss?", [])

        assert answer == _EMPTY_HISTORY_RESPONSE
        assert run_id is None

    @pytest.mark.asyncio
    async def test_run_id_none_when_run_tree_unavailable(
        self, mock_config: AppConfig, sample_history: list[dict[str, str]]
    ) -> None:
        """Test run_id is None when get_current_run_tree() returns None."""
        prompt_patch, llm_patch = self._patch_chain()
        with (
            prompt_patch,
            llm_patch,
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            _, run_id = await handle_conversational(
                mock_config, "What was my first question?", sample_history
            )

        assert run_id is None

    @pytest.mark.asyncio
    async def test_run_id_none_when_run_tree_raises(
        self, mock_config: AppConfig, sample_history: list[dict[str, str]]
    ) -> None:
        """Test run_id is None when get_current_run_tree() throws."""
        prompt_patch, llm_patch = self._patch_chain()
        with (
            prompt_patch,
            llm_patch,
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                side_effect=RuntimeError("Not in a trace"),
            ),
        ):
            answer, run_id = await handle_conversational(
                mock_config, "What was my first question?", sample_history
            )

        assert answer == "Your first question was about traceability."
        assert run_id is None


# ── stream_conversational_events ─────────────────────────────────────────────


class TestStreamConversationalEvents:
    """Tests for SSE streaming of conversational responses."""

    @pytest.mark.asyncio
    async def test_empty_history_streams_graceful_response(self, mock_config: AppConfig) -> None:
        """Test empty history yields token + done events with graceful message."""
        with patch(
            "requirements_graphrag_api.core.conversation.get_current_run_tree",
            return_value=None,
        ):
            events: list[str] = []
            async for event in stream_conversational_events(mock_config, "Recap", []):
                events.append(event)

        # Should have: event:token, data:token, event:done, data:done
        assert len(events) == 4
        assert StreamEventType.TOKEN.value in events[0]
        token_data = json.loads(events[1].split("data: ")[1].strip())
        assert token_data["token"] == _EMPTY_HISTORY_RESPONSE

        assert StreamEventType.DONE.value in events[2]
        done_data = json.loads(events[3].split("data: ")[1].strip())
        assert done_data["full_answer"] == _EMPTY_HISTORY_RESPONSE
        assert done_data["source_count"] == 0

    @pytest.mark.asyncio
    async def test_token_streaming_yields_individual_tokens(
        self,
        mock_config: AppConfig,
        sample_history: list[dict[str, str]],
        mock_llm: MagicMock,
    ) -> None:
        """Test true token streaming yields real-time tokens via astream."""
        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(
                mock_config, "What was my first question?", sample_history
            ):
                events.append(event)

        # Extract token data events (data lines that contain "token")
        token_contents: list[str] = []
        for event in events:
            if event.startswith("data: ") and '"token"' in event:
                data = json.loads(event[6:].strip())
                if "token" in data:
                    token_contents.append(data["token"])

        # Should have 6 individual tokens from mock_astream
        assert len(token_contents) == 6
        assert token_contents[0] == "Your "
        assert token_contents[-1] == "traceability."

    @pytest.mark.asyncio
    async def test_token_concatenation_equals_done_full_answer(
        self,
        mock_config: AppConfig,
        sample_history: list[dict[str, str]],
        mock_llm: MagicMock,
    ) -> None:
        """Test accumulated tokens match full_answer in done event."""
        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(
                mock_config, "What was my first question?", sample_history
            ):
                events.append(event)

        # Extract tokens
        tokens: list[str] = []
        full_answer = ""
        for event in events:
            if event.startswith("data: "):
                data = json.loads(event[6:].strip())
                if "token" in data:
                    tokens.append(data["token"])
                if "full_answer" in data:
                    full_answer = data["full_answer"]

        assert "".join(tokens) == full_answer

    @pytest.mark.asyncio
    async def test_done_event_has_run_id_when_available(
        self,
        mock_config: AppConfig,
        sample_history: list[dict[str, str]],
        mock_llm: MagicMock,
    ) -> None:
        """Test done event includes run_id from LangSmith run tree."""
        mock_run_tree = MagicMock()
        mock_run_tree.id = "streaming-run-id"

        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=mock_run_tree,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(
                mock_config, "What was my first question?", sample_history
            ):
                events.append(event)

        # Find the done data
        done_data = None
        for event in events:
            if event.startswith("data: "):
                data = json.loads(event[6:].strip())
                if "full_answer" in data:
                    done_data = data
                    break

        assert done_data is not None
        assert done_data["run_id"] == "streaming-run-id"

    @pytest.mark.asyncio
    async def test_done_event_no_run_id_when_unavailable(
        self,
        mock_config: AppConfig,
        sample_history: list[dict[str, str]],
        mock_llm: MagicMock,
    ) -> None:
        """Test done event omits run_id when not available."""
        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(mock_config, "Recap", sample_history):
                events.append(event)

        done_data = None
        for event in events:
            if event.startswith("data: "):
                data = json.loads(event[6:].strip())
                if "full_answer" in data:
                    done_data = data
                    break

        assert done_data is not None
        assert "run_id" not in done_data

    @pytest.mark.asyncio
    async def test_no_sources_or_phase_events_emitted(
        self,
        mock_config: AppConfig,
        sample_history: list[dict[str, str]],
        mock_llm: MagicMock,
    ) -> None:
        """Test conversational path emits no sources, cypher, or phase events."""
        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(
                mock_config, "What was my first question?", sample_history
            ):
                events.append(event)

        all_data = []
        for event in events:
            if event.startswith("data: "):
                data = json.loads(event[6:].strip())
                all_data.append(data)

        # None of the data payloads should have sources, cypher, or phase keys
        for data in all_data:
            assert "sources" not in data
            assert "cypher" not in data
            assert "phase" not in data

    @pytest.mark.asyncio
    async def test_streaming_error_yields_error_event(
        self, mock_config: AppConfig, sample_history: list[dict[str, str]]
    ) -> None:
        """Test that astream errors produce SSE error events."""
        mock_llm = MagicMock()

        async def failing_astream(*_args: Any, **_kwargs: Any):
            raise RuntimeError("LLM connection failed")
            yield  # make this an async generator

        mock_llm.astream = failing_astream

        with (
            patch("requirements_graphrag_api.core.conversation.ChatOpenAI", return_value=mock_llm),
            patch(
                "requirements_graphrag_api.core.conversation.get_current_run_tree",
                return_value=None,
            ),
        ):
            events: list[str] = []
            async for event in stream_conversational_events(mock_config, "Recap", sample_history):
                events.append(event)

        # Should have error event
        error_events = [e for e in events if StreamEventType.ERROR.value in e]
        assert len(error_events) > 0

        # Should have error data
        for event in events:
            if event.startswith("data: ") and '"error"' in event:
                data = json.loads(event[6:].strip())
                assert data["error"] == "Conversational response failed"
                break


# ── Prompt catalog ───────────────────────────────────────────────────────────


class TestConversationalPromptCatalog:
    """Tests for CONVERSATIONAL prompt registration in catalog."""

    def test_prompt_name_exists(self) -> None:
        assert PromptName.CONVERSATIONAL == "graphrag-conversational"

    def test_prompt_registered_in_definitions(self) -> None:
        assert PromptName.CONVERSATIONAL in PROMPT_DEFINITIONS

    def test_prompt_has_required_fields(self) -> None:
        defn = PROMPT_DEFINITIONS[PromptName.CONVERSATIONAL]
        assert defn.name == PromptName.CONVERSATIONAL
        assert defn.template is not None
        assert defn.metadata is not None
        assert defn.metadata.version == "1.0.0"

    def test_prompt_has_evaluation_criteria(self) -> None:
        defn = PROMPT_DEFINITIONS[PromptName.CONVERSATIONAL]
        assert "recall_accuracy" in defn.metadata.evaluation_criteria
        assert "summarization_quality" in defn.metadata.evaluation_criteria


# ── Checkpoint fallback ──────────────────────────────────────────────────────


class TestGetConversationHistoryFromCheckpoint:
    """Tests for checkpoint-based history retrieval."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_none_checkpoint(self) -> None:
        """Test aget_tuple returning None → empty list."""
        from requirements_graphrag_api.core.agentic.checkpoints import (
            get_conversation_history_from_checkpoint,
        )

        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple = AsyncMock(return_value=None)

        result = await get_conversation_history_from_checkpoint(mock_checkpointer, "thread-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_formatted_history_from_checkpoint(self) -> None:
        """Test valid checkpoint → formatted history list."""
        from langchain_core.messages import AIMessage, HumanMessage

        from requirements_graphrag_api.core.agentic.checkpoints import (
            get_conversation_history_from_checkpoint,
        )

        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="What is traceability?"),
                    AIMessage(content="Traceability is the ability to track requirements."),
                ]
            }
        }

        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple = AsyncMock(return_value=mock_checkpoint)

        result = await get_conversation_history_from_checkpoint(mock_checkpointer, "thread-123")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "What is traceability?"}
        assert result[1] == {
            "role": "assistant",
            "content": "Traceability is the ability to track requirements.",
        }

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self) -> None:
        """Test exception during checkpoint read → empty list."""
        from requirements_graphrag_api.core.agentic.checkpoints import (
            get_conversation_history_from_checkpoint,
        )

        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple = AsyncMock(
            side_effect=ConnectionError("Database unavailable")
        )

        result = await get_conversation_history_from_checkpoint(mock_checkpointer, "thread-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_raw_dict_messages(self) -> None:
        """Test checkpoint with raw dict messages (not LangChain objects)."""
        from requirements_graphrag_api.core.agentic.checkpoints import (
            get_conversation_history_from_checkpoint,
        )

        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "human", "content": "Hello"},
                    {"role": "ai", "content": "Hi there"},
                ]
            }
        }

        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple = AsyncMock(return_value=mock_checkpoint)

        result = await get_conversation_history_from_checkpoint(mock_checkpointer, "thread-123")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}
