# Serving image for the credit-risk API.
#
# Multi-stage on purpose. The build stage needs a compiler toolchain to resolve
# and install the wheels; the runtime stage needs none of it. Shipping gcc to
# production is both a larger image and a larger attack surface, and neither is
# needed to answer an HTTP request.

# ----------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build
# Copy only what the dependency resolution needs first, so an edit to source
# code does not invalidate the (slow) dependency layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

# ----------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# libgomp is LightGBM's OpenMP runtime. Without it the import fails at start-up
# with a linker error that reads like a missing Python package but is not one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. A container that does not need root should not have it.
RUN useradd --create-home --uid 10001 appuser

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

USER appuser
EXPOSE 8000

# Health is "a model is loaded and serving", not "the port accepts a connection".
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=10 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"model_loaded":true' || exit 1

CMD ["uvicorn", "credit_risk.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
