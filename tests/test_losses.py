import pytest
import torch

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
