#!/usr/bin/env python3
"""Reusable stdlib-only Python 3 MCP JSON-RPC 2.0 stdio client for Flint Studio E2E tests."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from typing import Any


def encode_frame(obj: dict[str, Any], framing: str = "cl") -> bytes:
    """Encode a JSON-RPC object as Content-Length ('cl') or NDJSON ('ndjson')."""
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if framing == "cl":
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        return header + payload
    if framing == "ndjson":
        return payload + b"\n"
    raise ValueError(f"Unsupported framing mode: {framing!r}")


def assert_byte0_clean(raw_stdout: bytes, framing: str = "auto") -> None:
    """Verify stdout starts cleanly at byte 0 with protocol framing and zero engine banners."""
    if not raw_stdout:
        raise AssertionError("MCP stdout is empty")
    if b"[Flint" in raw_stdout:
        raise AssertionError(f"Unexpected [Flint banner in MCP stdout: {raw_stdout[:200]!r}")
    if framing == "cl":
        if not raw_stdout.lower().startswith(b"content-length:"):
            raise AssertionError(f"Expected Content-Length at byte 0, got: {raw_stdout[:80]!r}")
    elif framing == "ndjson":
        if not raw_stdout.startswith(b"{"):
            raise AssertionError(f"Expected '{{' at byte 0 for NDJSON, got: {raw_stdout[:80]!r}")
    elif framing == "auto":
        if not (raw_stdout.lower().startswith(b"content-length:") or raw_stdout.startswith(b"{")):
            raise AssertionError(f"Invalid byte-0 MCP stdout: {raw_stdout[:80]!r}")
    else:
        raise ValueError(f"Unsupported framing mode: {framing!r}")


def parse_frames(buf: bytes, framing: str = "auto") -> list[dict[str, Any]]:
    """Parse one or more Content-Length or NDJSON frames from raw stdout bytes."""
    if not buf:
        return []
    assert_byte0_clean(buf, framing=framing)
    frames: list[dict[str, Any]] = []
    pos = 0
    n = len(buf)
    detected = framing
    if detected == "auto":
        detected = "cl" if buf[:15].lower() == b"content-length:" else "ndjson"

    if detected == "cl":
        while pos < n:
            while pos < n and buf[pos:pos + 1] in (b"\r", b"\n"):
                pos += 1
            if pos >= n:
                break
            sep_idx = buf.find(b"\r\n\r\n", pos)
            sep_len = 4
            alt_idx = buf.find(b"\n\n", pos)
            if alt_idx != -1 and (sep_idx == -1 or alt_idx < sep_idx):
                sep_idx = alt_idx
                sep_len = 2
            if sep_idx == -1:
                raise AssertionError(f"Malformed/unterminated Content-Length header at offset {pos}: {buf[pos:pos+100]!r}")
            header_bytes = buf[pos:sep_idx].decode("ascii", errors="strict")
            content_len = None
            for line in header_bytes.replace("\r\n", "\n").split("\n"):
                if ":" not in line:
                    raise AssertionError(f"Invalid header line: {line!r}")
                k, v = line.split(":", 1)
                if k.strip().lower() == "content-length":
                    content_len = int(v.strip())
            if content_len is None or content_len < 0:
                raise AssertionError(f"Missing or invalid Content-Length in header: {header_bytes!r}")
            body_start = sep_idx + sep_len
            body_end = body_start + content_len
            if body_end > n:
                raise AssertionError(f"Truncated Content-Length body: need {content_len}B, have {n - body_start}B")
            body = buf[body_start:body_end].decode("utf-8", errors="strict")
            frames.append(json.loads(body))
            pos = body_end
    elif detected == "ndjson":
        for raw_line in buf.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            frames.append(json.loads(stripped.decode("utf-8", errors="strict")))
    else:
        raise ValueError(f"Unsupported framing mode: {framing!r}")

    return frames


def run_batch(
    cmd: list[str],
    requests: list[dict[str, Any] | bytes],
    cwd: str | None = None,
    framing: str = "cl",
    timeout: float = 30.0,
    allow_stderr: bool = False,
) -> list[dict[str, Any]]:
    """Send a batch of JSON-RPC requests/raw bytes over stdin and parse all stdout frames."""
    stdin_chunks: list[bytes] = []
    for item in requests:
        if isinstance(item, bytes):
            stdin_chunks.append(item)
        else:
            stdin_chunks.append(encode_frame(item, framing=framing))
    stdin_data = b"".join(stdin_chunks)

    proc = subprocess.run(
        cmd,
        input=stdin_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"MCP process exited {proc.returncode}; stderr={proc.stderr.decode('utf-8', errors='replace')!r}"
        )
    if not allow_stderr and proc.stderr:
        raise AssertionError(f"Unexpected MCP stderr: {proc.stderr.decode('utf-8', errors='replace')!r}")
    if not proc.stdout:
        return []
    return parse_frames(proc.stdout, framing=framing)


MCP_E2E_PROTOCOL = "2025-06-18"
"""Revision the harness negotiates by default: the newest Flint speaks, so tool results carry
`structuredContent`. Clients on `2024-11-05` get only the text content block."""


def tool_payload(res: dict[str, Any]) -> Any:
    """The tool's JSON payload: `structuredContent` (2025-06-18) or the parsed text block (2024-11-05)."""
    sc = res.get("structuredContent")
    if sc is not None:
        return sc
    for block in res.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block.get("text", ""))
            except ValueError:
                return None
    return None


class McpStdioClient:
    """Interactive streaming MCP JSON-RPC 2.0 stdio client.

    `--mcp-stdio` refuses everything but `initialize`/`ping` until `initialize` (-32002), so with
    `auto_initialize` (the default) the first other request performs the handshake on 2025-06-18.
    Pass `auto_initialize=False` to exercise the pre-initialize lifecycle."""

    def __init__(
        self,
        cmd: list[str],
        cwd: str | None = None,
        framing: str = "cl",
        timeout: float = 30.0,
        allow_stderr: bool = False,
        auto_initialize: bool = True,
    ) -> None:
        if framing not in ("cl", "ndjson"):
            raise ValueError(f"framing must be 'cl' or 'ndjson', got {framing!r}")
        self.cmd = list(cmd)
        self.cwd = cwd
        self.framing = framing
        self.timeout = timeout
        self.allow_stderr = allow_stderr
        self.auto_initialize = auto_initialize
        self.initialized = False
        self.protocol_version: str | None = None
        self._next_id = 1
        self._buf = bytearray()
        self._raw_seen = bytearray()
        self._byte0_checked = False
        self._stdout_eof = False
        self._closed = False
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            bufsize=0,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.proc.stdout.fileno(), selectors.EVENT_READ)

    def __enter__(self) -> "McpStdioClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                if exc_type is None:
                    raise

    def _alloc_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send_raw(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _fill_once(self, wait_s: float) -> bool:
        assert self.proc.stdout is not None
        events = self._sel.select(timeout=max(0.0, wait_s))
        if not events:
            return False
        chunk = os.read(self.proc.stdout.fileno(), 4096)
        if not chunk:
            self._stdout_eof = True
            return False
        self._buf.extend(chunk)
        self._raw_seen.extend(chunk)
        if not self._byte0_checked and len(self._raw_seen) > 0:
            if b"[Flint" in self._raw_seen:
                raise AssertionError(f"Unexpected [Flint banner on stdout: {bytes(self._raw_seen[:200])!r}")
            if self.framing == "cl":
                prefix = b"content-length:"
                raw_lower = bytes(self._raw_seen).lower()
                if not prefix.startswith(raw_lower) and not raw_lower.startswith(prefix):
                    raise AssertionError(
                        f"Expected Content-Length at byte 0, got: {bytes(self._raw_seen[:80])!r}"
                    )
                if len(self._raw_seen) >= len(prefix):
                    assert_byte0_clean(bytes(self._raw_seen), framing="cl")
                    self._byte0_checked = True
            elif self.framing == "ndjson":
                assert_byte0_clean(bytes(self._raw_seen), framing="ndjson")
                self._byte0_checked = True
        return True

    def _try_pop_frame(self) -> dict[str, Any] | None:
        if self.framing == "cl":
            while self._buf[:1] in (b"\r", b"\n"):
                del self._buf[:1]
            if not self._buf:
                return None
            prefix = b"content-length:"
            buf_lower = bytes(self._buf).lower()
            if not prefix.startswith(buf_lower) and not buf_lower.startswith(prefix):
                raise AssertionError(
                    f"Expected Content-Length header, got: {bytes(self._buf[:80])!r}"
                )
            sep_idx = self._buf.find(b"\r\n\r\n")
            sep_len = 4
            alt_idx = self._buf.find(b"\n\n")
            if alt_idx != -1 and (sep_idx == -1 or alt_idx < sep_idx):
                sep_idx = alt_idx
                sep_len = 2
            if sep_idx == -1:
                return None
            header_bytes = bytes(self._buf[:sep_idx]).decode("ascii", errors="strict")
            content_len = None
            for line in header_bytes.replace("\r\n", "\n").split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip().lower() == "content-length":
                        content_len = int(v.strip())
            if content_len is None or content_len < 0:
                raise AssertionError(f"Missing Content-Length in header: {header_bytes!r}")
            body_start = sep_idx + sep_len
            body_end = body_start + content_len
            if len(self._buf) < body_end:
                return None
            body = bytes(self._buf[body_start:body_end]).decode("utf-8", errors="strict")
            del self._buf[:body_end]
            return json.loads(body)

        nl_idx = self._buf.find(b"\n")
        while nl_idx != -1:
            line = bytes(self._buf[:nl_idx]).strip()
            del self._buf[:nl_idx + 1]
            if line:
                if b"[Flint" in line:
                    raise AssertionError(f"Unexpected [Flint banner in NDJSON line: {line!r}")
                return json.loads(line.decode("utf-8", errors="strict"))
            nl_idx = self._buf.find(b"\n")
        return None

    def _read_frame(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            frame = self._try_pop_frame()
            if frame is not None:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for MCP {self.framing} response; partial_buf={bytes(self._buf[:200])!r}"
                )
            got = self._fill_once(min(0.25, remaining))
            if not got and self.proc.poll() is not None:
                frame = self._try_pop_frame()
                if frame is not None:
                    return frame
                stderr_tail = self.proc.stderr.read() if self.proc.stderr else b""
                raise EOFError(
                    f"MCP server exited ({self.proc.returncode}) before response; "
                    f"buf={bytes(self._buf)!r}, stderr={stderr_tail!r}"
                )

    def request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        self._ensure_initialized(method)
        rid = self._alloc_id()
        req: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self._send_raw(encode_frame(req, framing=self.framing))
        resp = self._read_frame(timeout)
        if resp.get("id") != rid:
            raise AssertionError(f"Mismatched JSON-RPC response id: expected {rid}, got {resp!r}")
        return resp

    def _ensure_initialized(self, method: str | None) -> None:
        if self.initialized or not self.auto_initialize or method in ("initialize", "ping"):
            return
        self.initialize()
        self.notify_initialized()

    def initialize(
        self,
        protocol_version: str = MCP_E2E_PROTOCOL,
        auth_token: str | None = None,
        client_name: str = "e2e-studio",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "1.0.0"},
        }
        if auth_token is not None:
            params["authToken"] = auth_token
        resp = self.request("initialize", params)
        if "error" in resp:
            raise AssertionError(f"initialize failed with error: {resp['error']!r}")
        self.initialized = True
        self.protocol_version = resp["result"].get("protocolVersion")
        return resp["result"]

    def structured(self) -> bool:
        """Whether the negotiated revision carries `structuredContent` (>= 2025-06-18)."""
        return (self.protocol_version or "") >= "2025-06-18"

    def _payload(self, name: str, res: dict[str, Any]) -> Any:
        sc = res.get("structuredContent")
        if self.structured():
            if not isinstance(sc, dict):
                raise AssertionError(f"tools/call({name}) missing structuredContent on {self.protocol_version}: {res!r}")
            return sc
        if sc is not None:
            raise AssertionError(f"tools/call({name}) sent structuredContent to a {self.protocol_version} client: {res!r}")
        return tool_payload(res)

    def notify_initialized(self, quiet_wait_s: float = 0.05) -> None:
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._send_raw(encode_frame(note, framing=self.framing))
        end_t = time.monotonic() + quiet_wait_s
        while time.monotonic() < end_t:
            if self._stdout_eof:
                break
            if self._fill_once(max(0.0, end_t - time.monotonic())):
                if self._buf:
                    raise AssertionError(
                        f"Notification must be silent, but server emitted stdout: {bytes(self._buf)!r}"
                    )
            elif self._stdout_eof:
                break

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self.request("tools/list", {})
        if "error" in resp:
            raise AssertionError(f"tools/list failed: {resp['error']!r}")
        tools = resp["result"].get("tools", [])
        if not isinstance(tools, list):
            raise AssertionError(f"tools/list result.tools is not a list: {resp!r}")
        return tools

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        # `timeout` overrides the client default for slow tools (project.package compiles the game).
        resp = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout)
        if "error" in resp:
            raise AssertionError(f"tools/call({name}) JSON-RPC error: {resp['error']!r}")
        res = resp.get("result", {})
        if res.get("isError", False):
            raise AssertionError(f"tools/call({name}) returned isError=True: {res!r}")
        sc = self._payload(name, res)
        if not isinstance(sc, dict) or sc.get("ok") is not True:
            raise AssertionError(f"tools/call({name}) structuredContent.ok != True: {res!r}")
        return sc

    def call_tool_expect_error(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        if "error" in resp:
            raise AssertionError(f"Expected tool error result, got protocol error: {resp['error']!r}")
        res = resp.get("result", {})
        sc = self._payload(name, res)
        if not res.get("isError", False):
            raise AssertionError(f"Expected isError=True for {name}, got: {res!r}")
        if not isinstance(sc, dict) or sc.get("ok") is not False:
            raise AssertionError(f"Expected structuredContent.ok=False for {name}, got: {res!r}")
        return sc

    def expect_protocol_error(self, payload: dict[str, Any] | bytes, expected_code: int) -> dict[str, Any]:
        self._ensure_initialized(None if isinstance(payload, bytes) else payload.get("method"))
        if isinstance(payload, bytes):
            self._send_raw(payload)
        else:
            self._send_raw(encode_frame(payload, framing=self.framing))
        resp = self._read_frame()
        err = resp.get("error")
        if not isinstance(err, dict) or err.get("code") != expected_code:
            raise AssertionError(f"Expected JSON-RPC error code {expected_code}, got: {resp!r}")
        return resp

    def close(self, timeout: float = 10.0) -> tuple[int, bytes]:
        if self._closed:
            return (self.proc.returncode or 0, b"")
        self._closed = True
        try:
            self._sel.close()
        except Exception:
            pass
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise
        stderr_bytes = self.proc.stderr.read() if self.proc.stderr is not None else b""
        if rc != 0:
            raise AssertionError(f"MCP server exited with code {rc}; stderr={stderr_bytes!r}")
        if not self.allow_stderr and stderr_bytes:
            raise AssertionError(f"MCP server wrote to stderr: {stderr_bytes!r}")
        return (rc, stderr_bytes)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Flint Studio MCP v2 stdio test client")
    parser.add_argument("--bin", required=True, help="Path to flint-studio binary")
    parser.add_argument("--cwd", default=".", help="Working directory for MCP server")
    parser.add_argument("--framing", choices=["cl", "ndjson"], default="cl")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra CLI args before --mcp-stdio")
    parser.add_argument("--call", default=None, help="Optional tool name to call after initialize")
    parser.add_argument("--args", default="{}", help="JSON arguments string for --call")
    parser.add_argument("--expect-tools", type=int, default=1, help="Minimum expected tool count")
    parser.add_argument("--timeout", type=float, default=30.0)
    ns = parser.parse_args(argv)

    cmd = [ns.bin, *ns.extra_arg, "--mcp-stdio"]
    with McpStdioClient(cmd, cwd=ns.cwd, framing=ns.framing, timeout=ns.timeout) as client:
        init_res = client.initialize()
        if init_res.get("protocolVersion") != MCP_E2E_PROTOCOL:
            raise AssertionError(f"Unexpected protocolVersion: {init_res!r}")
        client.notify_initialized()
        tools = client.list_tools()
        if len(tools) < ns.expect_tools:
            raise AssertionError(f"Expected >= {ns.expect_tools} tools, got {len(tools)}")
        if ns.call:
            tool_args = json.loads(ns.args)
            sc = client.call_tool(ns.call, tool_args)
            print(json.dumps(sc, sort_keys=True))
        else:
            print(json.dumps({"ok": True, "framing": ns.framing, "tool_count": len(tools)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except Exception as exc:
        print(f"[mcp_client] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
