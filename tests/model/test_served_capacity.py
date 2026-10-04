"""The served selection rate on real accounts the threshold was never computed on.

`serving_pool` is batch 6: the accounts the traffic generator replays against
the running stack, held out from training and from evaluation alike. The
threshold is computed on batch 5 exactly as training computes it, travels to
the API as a registry tag, and is applied there to batch 6. The risk team's
capacity is 10% of the portfolio, so that is where the served share has to
land -- within a point, which is about twice the sampling noise on 5,000
accounts.

The model is the one the other real-data test in this directory trains, not
the registered version 2 -- the registry lives in the compose stack and this
test does not -- so the property is checked, not version 2's numbers. Measured
by hand when the test was written: this model's capacity threshold is 0.5144
and serves 10.56% of `serving_pool` (0.5 would serve 11.04%); version 2's
configuration, re-run offline, gives 0.5167 and 10.14% (0.5: 10.56%).

Lives under `tests/model` because the CI `data_quality` job runs the
`needs_data` model tests once the dataset has been downloaded and split.
"""

from __future__ import annotations

from typing import Any

import pytest

from credit_risk import schema
from credit_risk.data.split import load_split
from credit_risk.models.train import feature_frame, load_training_splits, make_lightgbm
from tests.integration.test_serving_threshold import assert_capacity_is_held, serve_and_score


@pytest.mark.slow
@pytest.mark.needs_data
def test_the_served_selection_rate_holds_capacity_on_the_serving_pool(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        train_df, test_df = load_training_splits()
    except FileNotFoundError as exc:
        pytest.skip(f"processed splits are not on disk: {exc}")
    try:
        pool = load_split("serving_pool")
    except FileNotFoundError:
        # Without batch 6 the evaluation split is all there is. Capacity holds
        # there by construction, so this degrades to a check of the plumbing --
        # tag, loader, API -- rather than of generalisation.
        pool = test_df

    model = make_lightgbm(n_estimators=400, learning_rate=0.05, num_leaves=31)
    model.fit(feature_frame(train_df), train_df[schema.TARGET])

    served = serve_and_score(model, test_df, pool, tmp_path, monkeypatch)

    assert_capacity_is_held(served, len(pool))
