"""
models/meta_controller.py — Python-side MetaController (mirrors MQL5 MetaController.mqh).
Tracks per-agent Sharpe over a rolling window, computes softmax weights.
Used in live_monitor.py and for weighting ensemble signals.
"""

import math
import logging
from collections import deque
from typing import Dict, List

log = logging.getLogger(__name__)

AGENT_NAMES = ["PRISM", "GNN", "APEX", "CE"]
SHARPE_WINDOW = 252


class MetaController:
    """
    PnL-weighted bandit rebalancer.
    Tracks rolling Sharpe per agent → softmax → capital weights.
    """

    def __init__(self, n_agents: int = 4, window: int = SHARPE_WINDOW):
        self.n = n_agents
        self._window = window
        self._pnl_history: List[deque] = [deque(maxlen=window) for _ in range(n_agents)]
        self._weights = [1.0 / n_agents] * n_agents
        self._sharpes = [0.0] * n_agents

    def record_pnl(self, agent_id: int, pnl: float):
        if 0 <= agent_id < self.n:
            self._pnl_history[agent_id].append(pnl)

    def rebalance(self):
        for a in range(self.n):
            hist = list(self._pnl_history[a])
            if len(hist) < 5:
                self._sharpes[a] = 0.0
                continue
            import statistics
            mean = statistics.mean(hist)
            std  = statistics.stdev(hist) if len(hist) > 1 else 1e-10
            self._sharpes[a] = mean / (std + 1e-10) * math.sqrt(252)

        # Softmax on Sharpe values
        clipped = [max(-10.0, min(10.0, s)) for s in self._sharpes]
        exps = [math.exp(s) for s in clipped]
        total = sum(exps) or 1.0
        self._weights = [e / total for e in exps]

        log.info("MetaController rebalanced: %s",
                 {AGENT_NAMES[i]: f"{self._weights[i]:.3f}" for i in range(self.n)})

    def weight(self, agent_id: int) -> float:
        return self._weights[agent_id] if 0 <= agent_id < self.n else 0.25

    def sharpe(self, agent_id: int) -> float:
        return self._sharpes[agent_id] if 0 <= agent_id < self.n else 0.0

    def is_throttled(self, agent_id: int, threshold: float = 0.10) -> bool:
        return self.weight(agent_id) < threshold

    def weights_dict(self) -> Dict[str, float]:
        return {AGENT_NAMES[i]: self._weights[i] for i in range(self.n)}
