import pytest
import torch

from tristatelite.models.quantile_heads import BoundedQuantileHead, PositiveQuantileHead


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_quantile_heads_are_ordered_for_random_inputs(head_type):
    torch.manual_seed(7)

    output = head_type(16)(torch.randn(1000, 16))

    assert output.shape == (1000, 3)
    assert torch.all(output[:, 0] <= output[:, 1])
    assert torch.all(output[:, 1] <= output[:, 2])
    assert torch.all(output >= 0)


def test_bounded_head_stays_at_or_below_one_for_extreme_inputs():
    output = BoundedQuantileHead(8)(torch.randn(1000, 8) * 100)

    assert torch.all(output <= 1)


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_quantile_head_preserves_leading_dimensions_and_gradients(head_type):
    hidden = torch.randn(2, 5, 4, requires_grad=True)

    output = head_type(4)(hidden)
    output.sum().backward()

    assert output.shape == (2, 5, 3)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_quantile_head_rejects_non_positive_input_dimension(head_type):
    with pytest.raises(ValueError, match="positive"):
        head_type(0)
