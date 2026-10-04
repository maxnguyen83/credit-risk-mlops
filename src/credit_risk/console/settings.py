"""The console's configuration, read from the environment.

Kept apart from ``credit_risk.config.Settings`` on purpose. The console is a
separate container with its own environment, and none of these values mean
anything to training or serving; folding them into the shared settings would
put an Airflow password into the configuration of every process that imports
the package.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

FIVE_MEGABYTES = 5 * 1024 * 1024


class ConsoleSettings(BaseSettings):
    """Where Airflow is, where the shared data lives, and how the browser reaches the rest."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Airflow's stable REST API, reached over the compose network.
    airflow_url: str = "http://airflow:8080"
    airflow_username: str = ""
    airflow_password: SecretStr = SecretStr("")
    airflow_timeout_seconds: float = Field(default=8.0, gt=0)

    # The same folder Airflow mounts as its data directory, so a path written
    # into a DAG run's conf means the same file on both sides.
    console_data_dir: Path = Path("/app/data")

    # Links are opened by the browser, not by this container, so they use the
    # host's published ports rather than compose service names.
    console_link_host: str = "localhost"
    console_repo_url: str = "https://github.com/maxnguyen83/credit-risk-mlops"

    # 5,000 customers in the UCI layout is about 0.7 MB; 5 MB leaves room for
    # a larger month without letting a mistaken upload fill the disk.
    console_max_upload_bytes: int = Field(default=FIVE_MEGABYTES, gt=0)
