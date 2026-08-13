import numpy as np
import pytest
import torch

from tristatelite.losses.consistency import physics_consistency_loss
from tristatelite.losses.pinball import pinball_loss

QUANTILES = torch.tensor([0.05, 0.50, 0.95])


def test_pinball_is_zero_for_exact_prediction():
    target = torch.tensor([2.0, 4.0])
    prediction = target[:, None].expand(-1, 3)

    loss = pinball_loss(prediction, target, QUANTILES)

    assert loss.item() == 0.0


def test_pinball_matches_manual_underprediction():
    prediction = torch.zeros(1, 3)
    target = torch.tensor([2.0])

    loss = pinball_loss(prediction, target, QUANTILES)

    assert loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prediction", "target", "quantiles", "message"),
    [
        (torch.zeros(2, 2), torch.zeros(2), QUANTILES, "last dimension"),
        (torch.zeros(2, 3), torch.zeros(3), QUANTILES, "target shape"),
        (
            torch.zeros(2, 3),
            torch.zeros(2),
            torch.tensor([0.05, float("nan"), 0.95]),
            "finite",
        ),
        (
            torch.zeros(2, 3),
            torch.zeros(2),
            torch.tensor([0.50, 0.05, 0.95]),
            "increasing",
        ),
        (
            torch.zeros(2, 3),
            torch.zeros(2),
            torch.tensor([0.0, 0.50, 0.95]),
            "inside",
        ),
    ],
)
def test_pinball_rejects_invalid_shapes_and_quantiles(
    prediction, target, quantiles, message
):
    with pytest.raises(ValueError, match=message):
        pinball_loss(prediction, target, quantiles)


def test_pinball_accepts_a_trailing_singleton_target_dimension():
    prediction = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0]])

    loss = pinball_loss(prediction, target, QUANTILES)

    assert loss.item() == pytest.approx((0.05 + 0.05) / 3)


def test_consistency_is_zero_for_matching_physical_tte():
    soc = torch.tensor([0.5])
    soh = torch.tensor([0.8])
    q_ref = torch.tensor([2.0])
    current = torch.tensor([1.0])
    physical_tte = soc * soh * q_ref / current * 3600

    loss, diagnostics = physics_consistency_loss(
        soc,
        soh,
        torch.log1p(physical_tte),
        q_ref,
        current,
        torch.zeros(1),
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["active_fraction"].item() == 1.0
    assert diagnostics["mean_weight"].item() == 1.0
    assert diagnostics["median_physical_tte_s"].item() == pytest.approx(2880.0)


def test_low_current_is_masked_with_differentiable_zero():
    log_tte = torch.tensor([1.0], requires_grad=True)

    loss, diagnostics = physics_consistency_loss(
        torch.tensor([0.5]),
        torch.tensor([0.9]),
        log_tte,
        torch.tensor([2.0]),
        torch.tensor([0.01]),
        torch.tensor([0.0]),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert log_tte.grad is not None
    assert log_tte.grad.item() == 0.0
    assert diagnostics["active_fraction"].item() == 0.0
    assert diagnostics["mean_weight"].item() == 0.0
    assert diagnostics["median_physical_tte_s"].item() == 0.0


def test_nonfinite_physics_is_inactive():
    loss, diagnostics = physics_consistency_loss(
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.9, 0.9]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, float("nan")]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    )

    assert torch.isfinite(loss)
    assert diagnostics["active_fraction"].item() == 0.5


def test_variable_current_gets_lower_consistency_weight():
    common = (
        torch.tensor([0.5]),
        torch.tensor([0.9]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
    )

    _, stable = physics_consistency_loss(*common, torch.tensor([0.0]))
    _, variable = physics_consistency_loss(*common, torch.tensor([1.0]))

    assert variable["mean_weight"] < stable["mean_weight"]
    assert variable["mean_weight"].item() == pytest.approx(np.exp(-2.0))


def test_consistency_gradients_reach_all_predicted_medians():
    soc = torch.tensor([0.4], requires_grad=True)
    soh = torch.tensor([0.8], requires_grad=True)
    log_tte = torch.tensor([1.0], requires_grad=True)

    loss, _ = physics_consistency_loss(
        soc,
        soh,
        log_tte,
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
    )
    loss.backward()

    for value in (soc, soh, log_tte):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_consistency_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        physics_consistency_loss(
            torch.ones(2),
            torch.ones(1),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.zeros(2),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_current_a": 0.0}, "minimum_current_a"),
        ({"maximum_tte_s": 0.0}, "maximum_tte_s"),
    ],
)
def test_consistency_rejects_non_positive_thresholds(kwargs, message):
    values = torch.ones(1)
    with pytest.raises(ValueError, match=message):
        physics_consistency_loss(values, values, values, values, values, values, **kwargs)
