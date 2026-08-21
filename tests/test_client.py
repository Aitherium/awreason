"""Tests for awreason client.

No network calls, no live service. Use httpx MockTransport to validate
the wire contract.
"""

import httpx
import pytest
from awreason.client import (
    Depth,
    ReasoningClient,
    ReasoningError,
    SASEPhase,
    SessionInfo,
    Thought,
    ThoughtType,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


class MockTransport(httpx.BaseTransport):
    """Mock httpx transport for testing without a real service."""

    def __init__(self, responses: dict[str, dict]):
        """responses maps "METHOD /path" -> {"status": 200, "json": {...}}"""
        self.responses = responses
        self.calls = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle a request with mocked responses."""
        key = f"{request.method} {request.url.path}"
        self.calls.append((request.method, request.url.path, request.content))

        if key not in self.responses:
            return httpx.Response(404, json={"error": f"not mocked: {key}"})

        spec = self.responses[key]
        status = spec.get("status", 200)
        data = spec.get("json", {})
        return httpx.Response(status, json=data)


@pytest.fixture
def mock_transport():
    """Create a mock transport with standard responses."""
    return MockTransport(
        {
            "GET /health": {
                "status": 200,
                "json": {"status": "ok"},
            },
            "GET /stats": {
                "status": 200,
                "json": {
                    "sessions": 42,
                    "thoughts": 1234,
                    "average_depth": "gate",
                },
            },
            "POST /reason": {
                "status": 200,
                "json": {
                    "question": "why is the sky blue?",
                    "answer": "Light scattering in the atmosphere.",
                    "depth": "gate",
                },
            },
            "POST /sessions": {
                "status": 200,
                "json": {
                    "id": "sess-123",
                    "agent": "test-agent",
                    "query": "test query",
                    "status": "active",
                },
            },
            "GET /sessions": {
                "status": 200,
                "json": {
                    "sessions": [
                        {
                            "id": "sess-1",
                            "agent": "agent1",
                            "query": "q1",
                            "status": "completed",
                            "thought_count": 5,
                        },
                        {
                            "id": "sess-2",
                            "agent": "agent2",
                            "query": "q2",
                            "status": "active",
                            "thought_count": 2,
                        },
                    ]
                },
            },
            "GET /sessions/sess-123": {
                "status": 200,
                "json": {
                    "id": "sess-123",
                    "agent": "test-agent",
                    "query": "test query",
                    "status": "active",
                    "thought_count": 3,
                },
            },
            "GET /sessions/sess-123/tree": {
                "status": 200,
                "json": {
                    "id": "sess-123",
                    "thoughts": [
                        {
                            "id": "thought-1",
                            "content": "First thought",
                            "thought_type": "reasoning",
                        },
                        {
                            "id": "thought-2",
                            "content": "Second thought",
                            "thought_type": "analysis",
                        },
                    ],
                },
            },
            "POST /sessions/sess-123/end": {
                "status": 200,
                "json": {"status": "ended"},
            },
            "POST /thoughts": {
                "status": 200,
                "json": {"id": "thought-999"},
            },
            "GET /thoughts/recent": {
                "status": 200,
                "json": {
                    "thoughts": [
                        {
                            "id": "t1",
                            "session_id": "s1",
                            "content": "Recent thought",
                            "thought_type": "reasoning",
                        },
                    ]
                },
            },
            "GET /sase/capabilities": {
                "status": 200,
                "json": {
                    "phases": ["situation", "analysis", "synthesis", "execution"],
                    "depths": ["skip", "shallow", "gate", "deep", "critical"],
                },
            },
            "GET /gate/stats": {
                "status": 200,
                "json": {
                    "gated": 1234,
                    "reasoned": 567,
                    "skipped": 89,
                },
            },
        }
    )


def _make_client(transport: MockTransport) -> ReasoningClient:
    """Create a client with the mock transport."""
    c = ReasoningClient("http://localhost/")
    c._http = lambda: httpx.Client(transport=transport, base_url=c.base_url)
    return c


# ── Tests ──────────────────────────────────────────────────────────────────


class TestClientInit:
    """Test client initialization."""

    def test_default_base_url(self):
        """Default base URL is set correctly."""
        c = ReasoningClient()
        assert c.base_url == "http://127.0.0.1:8010"

    def test_custom_base_url(self):
        """Custom base URL is normalized."""
        c = ReasoningClient("http://localhost:8010/")
        assert c.base_url == "http://localhost:8010"

    def test_token_optional(self):
        """Token defaults to None."""
        c = ReasoningClient()
        assert c.token is None

    def test_token_from_arg(self):
        """Token can be provided."""
        c = ReasoningClient(token="test-token")
        assert c.token == "test-token"

    def test_timeout_default(self):
        """Timeout has a sensible default."""
        c = ReasoningClient()
        assert c.timeout == 120.0

    def test_timeout_custom(self):
        """Timeout can be customized."""
        c = ReasoningClient(timeout=30.0)
        assert c.timeout == 30.0


class TestReason:
    """Test the reason() method."""

    def test_simple_reason(self, mock_transport):
        """Ask a simple question."""
        c = _make_client(mock_transport)
        answer = c.reason("why is the sky blue?")
        assert answer == "Light scattering in the atmosphere."

    def test_reason_with_depth(self, mock_transport):
        """Ask with a specific depth."""
        c = _make_client(mock_transport)
        answer = c.reason("why is the sky blue?", depth="deep")
        assert answer is not None
        assert isinstance(answer, str)

    def test_reason_depth_enum(self, mock_transport):
        """Depth can be a Depth enum."""
        c = _make_client(mock_transport)
        answer = c.reason("test", depth=Depth.DEEP)
        assert answer is not None

    def test_reason_error_on_400(self, mock_transport):
        """Non-2xx status raises ReasoningError."""
        mock_transport.responses["POST /reason"]["status"] = 400
        mock_transport.responses["POST /reason"]["json"] = {"error": "bad request"}
        c = _make_client(mock_transport)

        with pytest.raises(ReasoningError):
            c.reason("test")

    def test_reason_error_on_500(self, mock_transport):
        """500 status raises ReasoningError."""
        mock_transport.responses["POST /reason"]["status"] = 500
        c = _make_client(mock_transport)

        with pytest.raises(ReasoningError):
            c.reason("test")

    def test_reason_error_on_no_answer(self, mock_transport):
        """No answer field raises ReasoningError."""
        mock_transport.responses["POST /reason"] = {
            "status": 200,
            "json": {},
        }
        c = _make_client(mock_transport)

        with pytest.raises(ReasoningError):
            c.reason("test")


class TestSessions:
    """Test session-related methods."""

    def test_list_sessions(self, mock_transport):
        """List all sessions."""
        c = _make_client(mock_transport)
        sessions = c.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].id == "sess-1"
        assert sessions[0].agent == "agent1"
        assert sessions[1].thought_count == 2

    def test_get_session(self, mock_transport):
        """Get one session."""
        c = _make_client(mock_transport)
        sess = c.get_session("sess-123")
        assert sess.id == "sess-123"
        assert sess.agent == "test-agent"
        assert sess.query == "test query"

    def test_get_session_tree(self, mock_transport):
        """Get the reasoning tree."""
        c = _make_client(mock_transport)
        tree = c.get_session_tree("sess-123")
        assert tree["id"] == "sess-123"
        assert len(tree["thoughts"]) == 2


class TestThoughts:
    """Test thought-related methods."""

    def test_recent_thoughts(self, mock_transport):
        """Get recent thoughts."""
        c = _make_client(mock_transport)
        thoughts = c.recent_thoughts(limit=10)
        assert len(thoughts) >= 1
        assert thoughts[0].id == "t1"
        assert thoughts[0].content == "Recent thought"

    def test_thought_dataclass(self):
        """Thought dataclass works."""
        t = Thought(
            id="t1",
            session_id="s1",
            content="test",
            thought_type="reasoning",
        )
        assert t.id == "t1"
        assert t.session_id == "s1"
        assert t.thought_type == "reasoning"


class TestServiceInfo:
    """Test service introspection."""

    def test_health(self, mock_transport):
        """Check service health."""
        c = _make_client(mock_transport)
        health = c.health()
        assert health["status"] == "ok"

    def test_stats(self, mock_transport):
        """Get service statistics."""
        c = _make_client(mock_transport)
        stats = c.stats()
        assert stats["sessions"] == 42
        assert stats["thoughts"] == 1234

    def test_sase_capabilities(self, mock_transport):
        """Get SASE capabilities."""
        c = _make_client(mock_transport)
        caps = c.sase_capabilities()
        assert "situation" in caps["phases"]
        assert "deep" in caps["depths"]

    def test_gate_stats(self, mock_transport):
        """Get gating statistics."""
        c = _make_client(mock_transport)
        stats = c.gate_stats()
        assert stats["gated"] == 1234
        assert stats["reasoned"] == 567


class TestEnums:
    """Test enumerations."""

    def test_sase_phase_values(self):
        """SASE phases have correct values."""
        assert SASEPhase.SITUATION.value == "situation"
        assert SASEPhase.ANALYSIS.value == "analysis"
        assert SASEPhase.SYNTHESIS.value == "synthesis"
        assert SASEPhase.EXECUTION.value == "execution"
        assert SASEPhase.COMPLETE.value == "complete"

    def test_depth_values(self):
        """Depth enum has correct values."""
        depths = {d.value for d in Depth}
        expected = {"skip", "shallow", "gate", "deep", "critical"}
        assert depths == expected

    def test_thought_type_values(self):
        """ThoughtType enum has correct values."""
        assert ThoughtType.REASONING.value == "reasoning"
        assert ThoughtType.OBSERVATION.value == "observation"
        assert ThoughtType.ANALYSIS.value == "analysis"


class TestSessionInfo:
    """Test SessionInfo dataclass."""

    def test_session_info_creation(self):
        """SessionInfo can be created."""
        info = SessionInfo(
            id="s1",
            agent="test",
            query="q",
            status="active",
            thought_count=5,
        )
        assert info.id == "s1"
        assert info.thought_count == 5

    def test_session_info_repr(self):
        """SessionInfo has a nice repr."""
        info = SessionInfo(
            id="s1",
            agent="test",
            query="q",
        )
        r = repr(info)
        assert "s1" in r
        assert "test" in r


class TestErrorHandling:
    """Test error handling."""

    def test_reasoning_error_is_exception(self):
        """ReasoningError is an Exception."""
        assert issubclass(ReasoningError, Exception)

    def test_reasoning_error_raised(self, mock_transport):
        """ReasoningError is raised on failure, not returned as empty."""
        mock_transport.responses["POST /reason"]["status"] = 500
        c = _make_client(mock_transport)

        with pytest.raises(ReasoningError):
            c.reason("test")

    def test_http_error_details(self, mock_transport):
        """HTTP errors include status code."""
        mock_transport.responses["POST /reason"]["status"] = 404
        c = _make_client(mock_transport)

        try:
            c.reason("test")
        except ReasoningError as e:
            assert "404" in str(e)


class TestPhrasesAndConstants:
    """Test SASE phases and depth constants."""

    def test_all_sase_phases_present(self):
        """All SASE phases are defined."""
        phases = {p.value for p in SASEPhase}
        required = {"situation", "analysis", "synthesis", "execution", "complete"}
        assert phases == required

    def test_all_thought_types_present(self):
        """Common thought types are defined."""
        types = {t.value for t in ThoughtType}
        assert "reasoning" in types
        assert "analysis" in types
        assert "tool_call" in types
        assert "error" in types

    def test_depth_gate_is_valid(self):
        """'gate' depth is a valid depth value."""
        # Verify it's a valid depth without needing a real service
        assert Depth.GATE.value == "gate"
        assert "gate" in {d.value for d in Depth}
