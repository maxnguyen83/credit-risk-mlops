"""Runtime configuration, read once from the environment.

Everything that differs between a laptop, CI and the compose stack lives
here. Modules import `settings`; they never read os.environ directly, so
there is exactly one place to look when a value is wrong.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- paths -------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # --- MLflow ------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="http://localhost:15020")
    mlflow_experiment: str = "credit-risk"
    model_name: str = "credit-risk"
    model_stage: str = "Production"

    # --- decision policy --------------------------------------------
    # `base` applies one threshold to everyone. `group_aware_equalized_odds`
    # applies the per-group thresholds fitted by the mitigation step -- which
    # is a deliberate, contestable choice, discussed in ETHICS.md.
    threshold_policy: Literal["base", "group_aware_equalized_odds"] = "base"
    # Where the base cutoff comes from. `registry` reads the `threshold_at_k`
    # tag that registration writes onto the model version -- the cutoff that
    # admits the intervention capacity on the evaluation split -- so a new
    # model brings its own cutoff with it. `env` ignores the tag and decides at
    # `decision_threshold`. A version without the tag (anything registered
    # before the tag existed) also decides at `decision_threshold`, and /health
    # reports that as `threshold_source: "fallback"` rather than hiding it.
    threshold_source: Literal["registry", "env"] = "registry"
    decision_threshold: float = 0.5

    # --- serving -----------------------------------------------------
    api_title: str = "Credit Default Early-Warning API"
    api_version: str = "1.0.0"
    git_sha: str = "unknown"
    max_batch_size: int = 1000
    risk_share_window: int = 200  # rolling window for the share gauges

    # --- intervention capacity --------------------------------------
    # The risk team can act on 10% of the portfolio per month. Every
    # business metric (recall@k, lift@k, expected loss) is defined against
    # this number rather than an arbitrary 0.5 cutoff.
    intervention_capacity_fraction: float = 0.10

    # --- cost matrix (NT$), for expected-loss reporting ---------------
    cost_false_negative: float = 30_000.0  # a default we failed to flag
    cost_false_positive: float = 500.0  # an unnecessary intervention

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}/{self.model_stage}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
