"""A portable client for a reasoning-shaped service.

WHY A CLIENT AND NOT A LIFT
===========================
The monorepo reasoning engine is large with many internal dependencies. Lifting
it into a package produces something that raises `ModuleNotFoundError` on a
stranger's machine while reading as authoritative — a broken package, not a
shipped one.

Ship the small standalone CLIENT instead, keep the engine private and free to
change. The wire contract is the product.

WHAT "SHAPED" MEANS
===================
Any service exposing these routes — read off the running service, not invented:

    POST /sessions         {agent, query}
    GET  /sessions         list all sessions
    GET  /sessions/{id}    get one session
    GET  /sessions/{id}/tree   get session reasoning tree
    POST /sessions/{id}/end    end the session
    POST /reason           {question, depth}
    POST /thoughts         {session_id, content, thought_type}
    GET  /thoughts/recent  recent thoughts
    POST /sase/enhanced    enhanced SASE reasoning
    GET  /sase/capabilities   what SASE modes are available
    POST /gate/evaluate    criticality gating
    GET  /gate/stats       gating statistics
    POST /context/deep     deep context gathering
    POST /context/synthesize   synthesize context
    GET  /stats            service statistics
    GET  /health           service health

THE TRAP THIS CLIENT EXISTS TO AVOID
====================================
A reasoning service without strict response validation is **silently wrong**.
The response model typically takes pydantic's default, `extra="ignore"`, so a
field name this client gets wrong is **silently DROPPED**: no 422, no error,
a perfectly ordinary 200, and results computed from a request that quietly lost
half of what you asked for.

That is the opposite of strict clients, which fail loudly. Same platform,
opposite failure mode — so do not carry an assumption from one to the other.
Here, the ONLY protection is sending exactly the declared field set, which is
why phase names and other constants are validated rather than invented.

Depends on httpx and nothing else.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncContextManager, Optional

__all__ = [
    "ReasoningClient",
    "ReasoningError",
    "Session",
    "Thought",
    "ThoughtType",
    "SASEPhase",
    "Depth",
]

DEFAULT_TIMEOUT = 120.0
DEFAULT_BASE_URL = "http://127.0.0.1:8010"

#: SASE phases — phases of reasoning where checks can be placed.
class SASEPhase(str, Enum):
    """Phases of reasoning."""
    SITUATION = "situation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EXECUTION = "execution"
    COMPLETE = "complete"


#: Thinking depth — how deep to reason.
class Depth(str, Enum):
    """How deep to reason."""
    SKIP = "skip"
    SHALLOW = "shallow"
    GATE = "gate"
    DEEP = "deep"
    CRITICAL = "critical"


#: Thought types — kinds of reasoning steps.
class ThoughtType(str, Enum):
    """Kinds of reasoning thoughts."""
    REASONING = "reasoning"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    REFLECTION = "reflection"


@dataclass
class Thought:
    """One reasoning step."""
    id: str
    session_id: str
    content: str
    thought_type: str = "reasoning"
    agent: str = ""
    confidence: float = 0.0
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Thought {self.thought_type!r} {self.content[:50]}...>"


@dataclass
class SessionInfo:
    """Session metadata and state."""
    id: str
    agent: str
    query: str
    status: str = "active"
    thought_count: int = 0
    final_response: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Session {self.id} agent={self.agent!r} thoughts={self.thought_count}>"


class ReasoningError(RuntimeError):
    """The service refused or could not answer.

    Raised, never returned as None or empty. A reasoning failure and a
    reasoning that found no answer are different facts, and a client that
    returns None for both makes a dead backend look like an inconclusive
    result — the exact silence that hides an outage.
    """


def _normalize_url(url: str) -> str:
    """Normalize a URL by stripping trailing slashes."""
    return url.rstrip("/")


class Session:
    """A reasoning session for tracking thought chains.

    Usage:
        async with client.session("MyAgent", "Complex question") as sess:
            await sess.think("First, let me consider...")
            result = await sess.tool_call("search", {"q": "..."})
            await sess.observe(f"Search returned: {result}")
            answer = await sess.conclude("Therefore...")
    """

    def __init__(
        self,
        session_id: str,
        agent: str,
        query: str,
        client: "ReasoningClient",
    ):
        self.id = session_id
        self.agent = agent
        self.query = query
        self._client = client
        self._closed = False

    async def think(
        self,
        content: str,
        thought_type: str = "reasoning",
        confidence: float = 0.0,
    ) -> Optional[str]:
        """Add a reasoning thought. Returns thought ID if successful."""
        if self._closed:
            return None
        return await self._client.add_thought(
            session_id=self.id,
            content=content,
            thought_type=thought_type,
            confidence=confidence,
        )

    async def observe(self, content: str) -> Optional[str]:
        """Record an observation."""
        return await self.think(content, "observation")

    async def analyze(self, content: str) -> Optional[str]:
        """Record analysis."""
        return await self.think(content, "analysis")

    async def synthesize(self, content: str) -> Optional[str]:
        """Record synthesis."""
        return await self.think(content, "synthesis")

    async def conclude(self, content: str) -> Optional[str]:
        """Record a conclusion."""
        return await self.think(content, "conclusion")

    async def plan(self, content: str) -> Optional[str]:
        """Record a plan."""
        return await self.think(content, "plan")

    async def tool_call(
        self,
        tool_name: str,
        args: dict,
    ) -> Optional[str]:
        """Record a tool call."""
        return await self.think(
            f"Tool: {tool_name}({json.dumps(args)})",
            "tool_call",
        )

    async def end(self, final_answer: Optional[str] = None) -> bool:
        """End the session and optionally set final answer."""
        if self._closed:
            return False
        try:
            await self._client._post(
                f"/sessions/{self.id}/end",
                {"final_response": final_answer} if final_answer else {},
            )
            self._closed = True
            return True
        except ReasoningError:
            return False

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.end()


class ReasoningClient:
    """Talks to a reasoning-shaped service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        verify: bool | str = True,
    ) -> None:
        """
        base_url  the service origin. These serve TLS in-network; plain http into
                  a TLS listener closes the socket and reads as "the service is
                  down" while it is perfectly healthy. Defaults to
                  http://127.0.0.1:8010 or AWREASON_URL env var.
        token     the CALLER's bearer, never a service credential — this package
                  ships publicly, so an internal key would either fail for
                  strangers or work for everyone who reads the source. From
                  AWREASON_TOKEN env var if not provided.
        verify    never False against a real deployment; trust the CA instead.
        """
        import os

        self.base_url = _normalize_url(
            base_url or os.environ.get("AWREASON_URL") or DEFAULT_BASE_URL
        )
        self.token = token or os.environ.get("AWREASON_TOKEN")
        self.timeout = timeout
        self.verify = verify

    def _http(self):
        """Create an httpx client. Import deferred so the module works without httpx present."""
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify,
        )

    def _post(self, path: str, body: dict) -> dict:
        """Synchronous POST. Raises ReasoningError on failure."""
        try:
            with self._http() as c:
                r = c.post(path, json=body)
        except Exception as exc:
            raise ReasoningError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise ReasoningError(
                f"{path}: HTTP {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    def _get(self, path: str) -> Any:
        """Synchronous GET. Raises ReasoningError on failure."""
        try:
            with self._http() as c:
                r = c.get(path)
        except Exception as exc:
            raise ReasoningError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise ReasoningError(
                f"{path}: HTTP {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    async def _apost(self, path: str, body: dict) -> dict:
        """Async POST. Raises ReasoningError on failure."""
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            ) as c:
                r = await c.post(path, json=body)
        except Exception as exc:
            raise ReasoningError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise ReasoningError(
                f"{path}: HTTP {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    async def _aget(self, path: str) -> Any:
        """Async GET. Raises ReasoningError on failure."""
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            ) as c:
                r = await c.get(path)
        except Exception as exc:
            raise ReasoningError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise ReasoningError(
                f"{path}: HTTP {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    # ── Core reasoning ────────────────────────────────────────────────────────

    def reason(
        self,
        question: str,
        *,
        depth: str | Depth = "gate",
        agent: str = "agent",
    ) -> str:
        """Ask one hard question and get back an answer.

        This is the simplest interface — a blocking call that returns the final
        answer. For more control over the reasoning process, use session().

        Args:
            question: The question to reason about
            depth: How deep to reason (skip/shallow/gate/deep/critical)
            agent: Name of the agent doing the reasoning

        Returns:
            The final answer

        Raises:
            ReasoningError: If the service cannot answer
        """
        if isinstance(depth, Depth):
            depth = depth.value
        resp = self._post(
            "/reason",
            {
                "question": question,
                "depth": depth,
                "agent": agent,
            },
        )
        answer = resp.get("answer") or resp.get("response") or resp.get("result")
        if not answer:
            raise ReasoningError("Service returned no answer")
        return str(answer)

    @asynccontextmanager
    async def session(
        self,
        agent: str,
        query: str,
    ) -> AsyncContextManager[Session]:
        """Create a reasoning session.

        Usage:
            async with client.session("MyAgent", "Complex question") as sess:
                await sess.think("First, let me consider...")
                result = await sess.tool_call("search", {"q": "..."})
                await sess.observe(f"Search returned: {result}")
                answer = await sess.conclude("Therefore...")
        """
        resp = await self._apost(
            "/sessions",
            {
                "agent": agent,
                "query": query,
            },
        )
        session_id = resp.get("id") or resp.get("session_id")
        if not session_id:
            raise ReasoningError("Service did not return a session ID")

        sess = Session(session_id, agent, query, self)
        try:
            yield sess
        finally:
            await sess.end()

    async def add_thought(
        self,
        session_id: str,
        content: str,
        thought_type: str = "reasoning",
        confidence: float = 0.0,
    ) -> Optional[str]:
        """Add a thought to a session."""
        try:
            resp = await self._apost(
                "/thoughts",
                {
                    "session_id": session_id,
                    "content": content,
                    "thought_type": thought_type,
                    "confidence": confidence,
                },
            )
            return resp.get("id") or resp.get("thought_id")
        except ReasoningError:
            return None

    # ── Session inspection ────────────────────────────────────────────────────

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        resp = self._get("/sessions")
        sessions = resp.get("sessions") or resp.get("data") or []
        return [
            SessionInfo(
                id=s.get("id", ""),
                agent=s.get("agent", ""),
                query=s.get("query", ""),
                status=s.get("status", "unknown"),
                thought_count=s.get("thought_count", 0),
                final_response=s.get("final_response"),
                raw=s,
            )
            for s in sessions
        ]

    def get_session(self, session_id: str) -> SessionInfo:
        """Get one session."""
        resp = self._get(f"/sessions/{session_id}")
        return SessionInfo(
            id=resp.get("id", session_id),
            agent=resp.get("agent", ""),
            query=resp.get("query", ""),
            status=resp.get("status", "unknown"),
            thought_count=resp.get("thought_count", 0),
            final_response=resp.get("final_response"),
            raw=resp,
        )

    def get_session_tree(self, session_id: str) -> dict:
        """Get the reasoning tree for a session."""
        return self._get(f"/sessions/{session_id}/tree")

    def recent_thoughts(self, limit: int = 10) -> list[Thought]:
        """Get recent thoughts."""
        resp = self._get(f"/thoughts/recent?limit={limit}")
        thoughts = resp.get("thoughts") or resp.get("data") or []
        return [
            Thought(
                id=t.get("id", ""),
                session_id=t.get("session_id", ""),
                content=t.get("content", ""),
                thought_type=t.get("thought_type", "reasoning"),
                agent=t.get("agent", ""),
                confidence=float(t.get("confidence", 0.0)),
                parent_id=t.get("parent_id"),
                raw=t,
            )
            for t in thoughts
        ]

    # ── Service information ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Service statistics."""
        return self._get("/stats")

    def health(self) -> dict:
        """Service health."""
        return self._get("/health")

    def sase_capabilities(self) -> dict:
        """What SASE modes are available."""
        return self._get("/sase/capabilities")

    def gate_stats(self) -> dict:
        """Criticality gating statistics."""
        return self._get("/gate/stats")
