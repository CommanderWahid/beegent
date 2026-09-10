"""Run discover() in a background thread and stream its log lines."""

import json
import logging
import threading
import uuid

from beegent.run import discover, log_summary


class _Lines:
    """A run's log, RETAINED: a queue drains, so a reload would lose everything streamed."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.done = False
        self._cv = threading.Condition()

    def add(self, line: str) -> None:
        with self._cv:
            self.lines.append(line)
            self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self.done = True
            self._cv.notify_all()

    def tail(self):
        """Replay from the top, then block for more - by index, so readers never split a log."""
        seen = 0
        while True:
            with self._cv:
                while seen >= len(self.lines) and not self.done:
                    self._cv.wait()
                if seen >= len(self.lines):
                    return
                fresh, seen = self.lines[seen:], len(self.lines)
            yield from fresh


class _Tap(logging.Handler):
    """Every _log.info in the pipeline already IS the progress feed - tap it, add nothing."""

    def __init__(self, sink: _Lines):
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.add(record.getMessage())


class Runner:
    """ONE run at a time. discover() interleaves store writes and Ollama serves one call."""

    def __init__(self, out: str = "candidate_list.json"):
        self.out = out
        self._lock = threading.Lock()
        self._lines: dict[str, _Lines] = {}
        self._results: dict[str, dict] = {}
        self._stops: dict[str, threading.Event] = {}

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def start(self, country: str, use_case: str) -> str | None:
        """None when a run is already in flight - the caller turns that into a 409."""
        if not self._lock.acquire(blocking=False):
            return None
        run_id = str(uuid.uuid4())
        self._lines[run_id] = _Lines()
        self._stops[run_id] = threading.Event()
        threading.Thread(target=self._work, args=(run_id, country, use_case),
                         daemon=True).start()
        return run_id

    def result(self, run_id: str) -> dict | None:
        return self._results.get(run_id)

    def stop(self, run_id: str) -> bool:
        """Cooperative: discover() checks this between angles and steps, and exits clean."""
        event = self._stops.get(run_id)
        if event is None:
            return False
        event.set()
        return True

    def stream(self, run_id: str):
        """Blocks on new lines, so a slow step costs nothing and loses no line."""
        sink = self._lines.get(run_id)
        if sink is None:
            return
        yield from sink.tail()

    def _work(self, run_id: str, country: str, use_case: str) -> None:
        sink = self._lines[run_id]
        # The whole package, so pipeline stages are tapped too, not just run.py.
        log = logging.getLogger("beegent")
        tap = _Tap(sink)
        log.addHandler(tap)
        # Set the LEVEL too: served by `uvicorn api.main:app`, main() never runs, so the
        # logger sits at WARNING and drops every _log.info before the tap ever sees it.
        was = log.level
        log.setLevel(logging.INFO)
        try:
            run = discover(country, use_case, should_stop=self._stops[run_id].is_set)
            # The same block the CLI prints, so the Logs tab ends with what it cost.
            log_summary(run, log=sink.add)
            self._results[run_id] = run.to_dict()
            # candidate_list.json is the run's DELIVERABLE, written however it was started.
            with open(self.out, "w", encoding="utf-8") as fh:
                json.dump(run.to_dict(), fh, indent=2, ensure_ascii=False)
        except Exception as exc:  # discover() fails soft, so this is belt-and-braces
            sink.add(f"[error] {type(exc).__name__}: {exc}")
            self._results[run_id] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            log.setLevel(was)
            log.removeHandler(tap)
            sink.finish()
            self._lock.release()
