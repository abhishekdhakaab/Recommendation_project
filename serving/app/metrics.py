"""Lightweight Prometheus-style metrics for the serving layer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ServingMetrics:
    """In-memory counters and latency sums for local serving."""

    recommend_requests_total: int = 0
    feedback_requests_total: int = 0
    recommend_latency_ms_sum: float = 0.0
    stage_latency_ms_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record_recommendation(self, latency_ms: dict[str, float]) -> None:
        self.recommend_requests_total += 1
        self.recommend_latency_ms_sum += float(latency_ms.get("total", 0.0))
        for stage, value in latency_ms.items():
            self.stage_latency_ms_sum[stage] += float(value)

    def record_feedback(self) -> None:
        self.feedback_requests_total += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP seqrec_recommend_requests_total Total recommendation requests.",
            "# TYPE seqrec_recommend_requests_total counter",
            f"seqrec_recommend_requests_total {self.recommend_requests_total}",
            "# HELP seqrec_feedback_requests_total Total feedback requests.",
            "# TYPE seqrec_feedback_requests_total counter",
            f"seqrec_feedback_requests_total {self.feedback_requests_total}",
            "# HELP seqrec_recommend_latency_ms_sum Sum of recommendation total latency in milliseconds.",
            "# TYPE seqrec_recommend_latency_ms_sum counter",
            f"seqrec_recommend_latency_ms_sum {self.recommend_latency_ms_sum}",
            "# HELP seqrec_stage_latency_ms_sum Sum of per-stage latency in milliseconds.",
            "# TYPE seqrec_stage_latency_ms_sum counter",
        ]
        for stage in sorted(self.stage_latency_ms_sum):
            lines.append(f'seqrec_stage_latency_ms_sum{{stage="{stage}"}} {self.stage_latency_ms_sum[stage]}')
        return "\n".join(lines) + "\n"
