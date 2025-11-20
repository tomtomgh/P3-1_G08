# strategies/manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional
import pandas as pd

from .Utils import StrategyWindow
from .Fighting import FightingConfig, detect_fighting
from .Carefulness import CarefulnessConfig, detect_carefulness
from .TrialError import TrialErrorConfig, detect_trial_error

StrategyDetector = Callable[[pd.DataFrame, Any], List[StrategyWindow]]


@dataclass
class StrategySpec:
    name: str
    detector: StrategyDetector
    config: Any
    enabled: bool = True


class StrategyManager:
    """Register and run strategy detectors."""
    def __init__(self, strategies: Optional[Mapping[str, StrategySpec]] = None) -> None:
        self._strategies: Dict[str, StrategySpec] = dict(strategies or {})

    def register(self, name: str, detector: StrategyDetector, config: Any, enabled: bool = True) -> None:
        self._strategies[name] = StrategySpec(name=name, detector=detector, config=config, enabled=enabled)

    def enable(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = False

    def list_strategies(self) -> List[str]:
        return list(self._strategies.keys())

    def enabled_strategies(self) -> List[str]:
        return [name for name, spec in self._strategies.items() if spec.enabled]

    def run_one(self, name: str, df: pd.DataFrame) -> List[StrategyWindow]:
        spec = self._strategies.get(name)
        if spec is None or not spec.enabled:
            return []
        return spec.detector(df, spec.config)

    def run_all(self, df: pd.DataFrame) -> Dict[str, List[StrategyWindow]]:
        results: Dict[str, List[StrategyWindow]] = {}
        for name, spec in self._strategies.items():
            if not spec.enabled:
                continue
            results[name] = spec.detector(df, spec.config)
        return results

    @classmethod
    def default(
        cls,
        fighting_cfg: Optional[FightingConfig] = None,
        carefulness_cfg: Optional[CarefulnessConfig] = None,
        trial_error_cfg: Optional[TrialErrorConfig] = None,
    ) -> "StrategyManager":
        strategies: Dict[str, StrategySpec] = {}
        if fighting_cfg is not None:
            strategies["fighting"] = StrategySpec("fighting", detect_fighting, fighting_cfg, True)
        if carefulness_cfg is not None:
            strategies["carefulness"] = StrategySpec("carefulness", detect_carefulness, carefulness_cfg, True)
        if trial_error_cfg is not None:
            strategies["trial_error"] = StrategySpec("trial_error", detect_trial_error, trial_error_cfg, True)
        return cls(strategies)

# Public API
__all__ = ["StrategyDetector", "StrategySpec", "StrategyManager"]
