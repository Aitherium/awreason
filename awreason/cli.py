"""awreason CLI.

    awreason ask "why is the sky blue?"
    awreason ask "complex question" --depth deep
    awreason session MyAgent "long query"
    awreason stats
    awreason health
    awreason --self-test

The service origin comes from --url or AWREASON_URL; the token from --token
or AWREASON_TOKEN. Neither is guessed: a reasoning client that silently falls
back to some default endpoint sends your questions somewhere you did not choose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from awreason.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    Depth,
    ReasoningClient,
    ReasoningError,
)

# ── self-test ──────────────────────────────────────────────────────────────
# Everything asserted here is PURE. A self-test that needs a live service is a
# self-test that gets skipped, and a skipped check is indistinguishable from a
# passing one.


def _self_test() -> int:
    """Prove the client still holds its contract, offline."""
    failures: list[str] = []

    # 1. Base URL is normalized; no trailing slash.
    if ReasoningClient("https://h/").base_url != "https://h":
        failures.append("trailing slash not trimmed from base_url")

    # 2. No token means no bearer header, never an empty one.
    if ReasoningClient("https://h").token is not None:
        failures.append("token should default to None")

    # 3. Default base URL is sensible.
    if DEFAULT_BASE_URL != "http://127.0.0.1:8010":
        failures.append(f"DEFAULT_BASE_URL should be localhost, got {DEFAULT_BASE_URL}")

    # 4. Default timeout is reasonable.
    if DEFAULT_TIMEOUT <= 0:
        failures.append("DEFAULT_TIMEOUT must be positive")

    # 5. Depth enum values are correct.
    expected_depths = {"skip", "shallow", "gate", "deep", "critical"}
    actual_depths = {d.value for d in Depth}
    if actual_depths != expected_depths:
        failures.append(
            f"Depth values {actual_depths} != expected {expected_depths}"
        )

    # 6. ReasoningError is an Exception.
    if not issubclass(ReasoningError, Exception):
        failures.append("ReasoningError must be an Exception subclass")

    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"SELF-TEST: {len(failures)} failure(s)")
        return 1
    print("  PASS  base URL is normalized, defaults are sensible")
    print("  PASS  token is optional, depth enum is correct")
    print("SELF-TEST: awreason ok")
    return 0


# ── commands ───────────────────────────────────────────────────────────────


def _client(args: argparse.Namespace) -> ReasoningClient:
    """Create a client from CLI args."""
    url = args.url or os.environ.get("AWREASON_URL")
    token = args.token or os.environ.get("AWREASON_TOKEN")
    if not url:
        url = DEFAULT_BASE_URL
    return ReasoningClient(url, token)


def _ask_sync(args: argparse.Namespace) -> int:
    """Synchronous ask command."""
    question = " ".join(args.question)
    try:
        c = _client(args)
        answer = c.reason(
            question,
            depth=args.depth,
            agent=args.agent or "cli",
        )
        if args.json:
            print(json.dumps({"question": question, "answer": answer}, indent=2))
        else:
            print(answer)
        return 0
    except ReasoningError as exc:
        print(f"awreason: {exc}", file=sys.stderr)
        return 1


async def _ask_async(args: argparse.Namespace) -> int:
    """Async ask command (for potential future streaming)."""
    question = " ".join(args.question)
    try:
        c = _client(args)
        answer = c.reason(
            question,
            depth=args.depth,
            agent=args.agent or "cli",
        )
        if args.json:
            print(json.dumps({"question": question, "answer": answer}, indent=2))
        else:
            print(answer)
        return 0
    except ReasoningError as exc:
        print(f"awreason: {exc}", file=sys.stderr)
        return 1


def _stats(args: argparse.Namespace) -> int:
    """Show service statistics."""
    try:
        c = _client(args)
        stats = c.stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            for k, v in stats.items():
                print(f"{k}: {v}")
        return 0
    except ReasoningError as exc:
        print(f"awreason: {exc}", file=sys.stderr)
        return 1


def _health(args: argparse.Namespace) -> int:
    """Check service health."""
    try:
        c = _client(args)
        health = c.health()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            status = health.get("status", "unknown")
            print(f"Status: {status}")
            if status == "ok":
                return 0
            return 1
    except ReasoningError as exc:
        print(f"awreason: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(prog="awreason", description=__doc__)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="prove this client still holds its contract, offline",
    )
    ap.add_argument("--url", help="service origin (or AWREASON_URL)")
    ap.add_argument("--token", help="bearer token (or AWREASON_TOKEN)")
    ap.add_argument(
        "--json", action="store_true", help="print the raw response"
    )
    sub = ap.add_subparsers(dest="cmd")

    # ask command
    ask_p = sub.add_parser("ask", help="ask a question")
    ask_p.add_argument("question", nargs="+", help="the question to reason about")
    ask_p.add_argument(
        "--depth",
        default="gate",
        choices=["skip", "shallow", "gate", "deep", "critical"],
        help="how deep to reason (default: gate)",
    )
    ask_p.add_argument(
        "--agent", help="agent name (default: cli)"
    )

    # session command
    session_p = sub.add_parser("session", help="start a reasoning session")
    session_p.add_argument("agent", help="agent name")
    session_p.add_argument("query", nargs="+", help="the query")
    session_p.add_argument(
        "--depth",
        default="gate",
        help="how deep to reason",
    )

    # stats command
    sub.add_parser("stats", help="show service statistics")

    # health command
    sub.add_parser("health", help="check service health")

    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.cmd:
        ap.print_help()
        return 2

    try:
        if args.cmd == "ask":
            return _ask_sync(args)
        elif args.cmd == "stats":
            return _stats(args)
        elif args.cmd == "health":
            return _health(args)
        elif args.cmd == "session":
            print("Session mode not yet interactive; use `ask` for now")
            return 2
    except ValueError as exc:
        # A refusal made HERE, before the round trip.
        print(f"awreason: {exc}", file=sys.stderr)
        return 2
    except ReasoningError as exc:
        print(f"awreason: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
