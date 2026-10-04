# Architecture Decision Records

One file per decision that a reasonable engineer could have made differently.
Each records the context, what we chose, what we rejected, why, and the price.

| # | Decision |
|---|---|
| [0001](./0001-modular-monolith-api.md) | Modular monolith for the API, not microservices |
| [0002](./0002-batch-not-streaming.md) | Batch scoring, not streaming |
| [0003](./0003-registry-as-source-of-truth.md) | MLflow Registry is the source of truth for the served model |
| [0004](./0004-postgres-objectstore-behind-mlflow.md) | Postgres and an S3 object store (SeaweedFS) behind MLflow, not SQLite and a folder |
| [0005](./0005-airflow-standalone.md) | Airflow standalone in one container, not CeleryExecutor |
| [0006](./0006-two-interpreters-in-airflow-image.md) | Two Python interpreters inside the Airflow image |
| [0007](./0007-lightgbm-with-logistic-baseline.md) | LightGBM as the candidate, logistic regression as the baseline |
| [0008](./0008-github-hosted-ci.md) | CI on GitHub-hosted runners, CD on a self-hosted runner on the stack host |
| [0009](./0009-fairness-gate-blocks-registration.md) | Fairness thresholds block model registration |
| [0010](./0010-multistage-nonroot-image.md) | Multi-stage Dockerfile with a non-root runtime |

The full trade-off narrative lives in [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
section 4; these files exist so a single decision can be cited, linked and
revisited on its own.
