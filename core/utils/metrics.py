import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class Metrics:
    """
    Thread-safe in-process metrics: per-operation request counts, error
    counts, and latency totals. Good enough for a single-process service;
    swap for Prometheus/StatsD if you scale out.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._latency_ms_total: dict[str, float] = defaultdict(float)
        self._started_at = time.time()

    def record(self, operation: str, duration_ms: float, error: bool = False) -> None:
        with self._lock:
            self._counts[operation] += 1
            self._latency_ms_total[operation] += duration_ms
            if error:
                self._errors[operation] += 1

    @contextmanager
    def timer(self, operation: str):
        """Context manager that records latency and error status for a block."""
        start = time.perf_counter()
        errored = False
        try:
            yield
        except Exception:
            errored = True
            raise
        finally:
            self.record(operation, (time.perf_counter() - start) * 1000, error=errored)

    def snapshot(self) -> dict:
        with self._lock:
            operations = {}
            for op, count in self._counts.items():
                total_ms = self._latency_ms_total[op]
                operations[op] = {
                    "count":          count,
                    "errors":         self._errors[op],
                    "avg_latency_ms": round(total_ms / count, 2) if count else 0.0,
                }
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "total_requests": sum(self._counts.values()),
                "total_errors":   sum(self._errors.values()),
                "operations":     operations,
            }

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._errors.clear()
            self._latency_ms_total.clear()
            self._started_at = time.time()


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the process-wide Metrics singleton."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics
