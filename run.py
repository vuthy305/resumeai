"""Start BOTH websites with one command:

    python run.py

    Careers site (public, candidates)  ->  http://localhost:8000
    HR portal (internal, HR staff)     ->  http://localhost:8001   (API docs: /docs)

Press Ctrl+C once to stop both. Ports: RESUMEAI_CAREERS_PORT / RESUMEAI_HR_PORT.
You can also run them in two terminals:
    uvicorn api.careers_app:app --port 8000
    uvicorn api.main:app --port 8001
"""
import src  # noqa: F401,I001  (Windows OpenMP fix before torch/paddle load)
import asyncio
import signal

import uvicorn

from src.config import CAREERS_PORT, HR_PORT


class _Server(uvicorn.Server):
    """Two servers share one process, so Ctrl+C is handled once, below, for both."""
    def install_signal_handlers(self):
        pass

    def capture_signals(self):  # uvicorn >= 0.29
        import contextlib
        return contextlib.nullcontext()


async def main():
    servers = [_Server(uvicorn.Config("api.careers_app:app", host="127.0.0.1", port=CAREERS_PORT, log_level="info")),
               _Server(uvicorn.Config("api.main:app", host="127.0.0.1", port=HR_PORT, log_level="info"))]

    def stop(*_):
        for s in servers:
            s.should_exit = True
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    print(f"\n  Careers site (candidates):  http://localhost:{CAREERS_PORT}\n"
          f"  HR portal (HR staff):       http://localhost:{HR_PORT}\n"
          f"  API docs (HR only):         http://localhost:{HR_PORT}/docs\n  Press Ctrl+C to stop.\n")
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
