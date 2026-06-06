from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


class EngineError(RuntimeError):
    pass


class PikafishClient:
    def __init__(
        self,
        engine_path: str,
        cwd: str | Path | None = None,
        eval_file_path: str | None = None,
        threads: int | None = None,
        hash_mb: int | None = None,
    ) -> None:
        self.engine_path = self._resolve_path(engine_path, cwd)
        self.eval_file_path = self._resolve_path(eval_file_path, cwd) if eval_file_path else None
        self.threads = threads
        self.hash_mb = hash_mb
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return

        if not self.engine_path.exists():
            raise EngineError(
                f"Pikafish 引擎不存在：{self.engine_path}。请在 config.json 中配置 pikafish_path。"
            )

        if self.eval_file_path and not self.eval_file_path.exists():
            raise EngineError(
                f"Pikafish 缺少神经网络文件：{self.eval_file_path}\n"
                "请下载 pikafish.nnue，并放到 engines 目录，或在 config.json 中修改 pikafish_eval_file。"
            )

        try:
            self.process = subprocess.Popen(
                [str(self.engine_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.engine_path.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise EngineError(f"无法启动 Pikafish：{exc}") from exc

        self._send("uci")
        self._wait_for("uciok", timeout_seconds=10)
        if self.threads and self.threads > 0:
            self._send(f"setoption name Threads value {int(self.threads)}")
        if self.hash_mb and self.hash_mb > 0:
            self._send(f"setoption name Hash value {int(self.hash_mb)}")
        if self.eval_file_path:
            self._send(f"setoption name EvalFile value {self.eval_file_path}")
        self._send("isready")
        self._wait_for("readyok", timeout_seconds=10)

    def analyze(
        self,
        fen: str,
        movetime_ms: int = 1000,
        depth: int | None = None,
        mode: str = "movetime",
    ) -> tuple[str, float]:
        self.start()
        if not self.process or not self.process.stdout:
            raise EngineError("Pikafish 进程未正确启动。")

        start_time = time.perf_counter()
        self._send(f"position fen {fen}")
        if mode == "depth" and depth and depth > 0:
            self._send(f"go depth {int(depth)}")
            timeout_seconds = max(10, int(depth) * 3)
        else:
            self._send(f"go movetime {int(movetime_ms)}")
            timeout_seconds = max(5, movetime_ms / 1000 + 8)

        bestmove_line = self._wait_for("bestmove", timeout_seconds=timeout_seconds)
        elapsed = time.perf_counter() - start_time
        return bestmove_line.strip(), elapsed

    def close(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
        self.process = None

    def _send(self, command: str) -> None:
        if not self.process or not self.process.stdin:
            raise EngineError("Pikafish 进程不可用。")
        try:
            self.process.stdin.write(command + os.linesep)
            self.process.stdin.flush()
        except OSError as exc:
            raise EngineError(f"向 Pikafish 发送命令失败：{command}") from exc

    def _wait_for(self, marker: str, timeout_seconds: float) -> str:
        if not self.process or not self.process.stdout:
            raise EngineError("Pikafish 进程输出不可用。")

        deadline = time.monotonic() + timeout_seconds
        lines: list[str] = []
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = "".join(lines).strip()
                raise EngineError(f"Pikafish 已退出。最近输出：{output}")
            line = self.process.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            lines.append(line)
            if line.startswith(marker) or marker in line:
                return line

        output = "".join(lines).strip()
        raise EngineError(f"等待 Pikafish 返回 {marker} 超时。最近输出：{output}")

    @staticmethod
    def _resolve_path(path_text: str | None, cwd: str | Path | None) -> Path:
        if not path_text:
            raise ValueError("path_text must not be empty")
        path = Path(path_text)
        if path.is_absolute():
            return path
        base = Path(cwd) if cwd else Path(__file__).resolve().parent
        return (base / path).resolve()
