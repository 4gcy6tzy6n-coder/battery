"""Contract tests for the TriStateLite joint probabilistic network."""

from __future__ import annotations

import pytest
import torch

from tristatelite.experiments.model import PointHead, TriStateLiteNet


def _toy_batch(batch: int, length: int, fast_dim: int, slow_dim: int) -> dict[str, torch.Tensor]:
    fast_x = torch.randn(batch, length, fast_dim)
    fast_mask = torch.zeros(batch, length, dtype=torch.bool)
    for i in range(batch):
        m = int((i % length) + 1)
        fast_mask[i, length - m :] = True
    slow_x = torch.randn(batch, 6, slow_dim)
    slow_mask = torch.ones(batch, 6, dtype=torch.bool)
    return {"fast_x": fast_x, "fast_mask": fast_mask, "slow_x": slow_x, "slow_mask": slow_mask}


def test_net_outputs_three_ordered_quantile_triples() -> None:
    torch.manual_seed(0)
    model = TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=1)
    batch = _toy_batch(8, 128, 12, 8)
    soc, soh, tte = model(**batch)
    assert soc.shape == (8, 3)
    assert soh.shape == (8, 3)
    assert tte.shape == (8, 3)
    for out in (soc, soh):
        assert torch.all(out[:, 0] <= out[:, 1]) and torch.all(out[:, 1] <= out[:, 2])
        assert torch.all((out >= 0) & (out <= 1))
    assert torch.all(tte >= 0)
    assert torch.all(tte[:, 0] <= tte[:, 1]) and torch.all(tte[:, 1] <= tte[:, 2])


def test_net_gradients_flow_and_optimizer_steps() -> None:
    torch.manual_seed(1)
    model = TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _toy_batch(4, 64, 12, 8)
    before = next(model.parameters()).clone()
    soc, soh, tte = model(**batch)
    (soc.sum() + soh.sum() + tte.sum()).backward()
    opt.step()
    after = next(model.parameters())
    assert not torch.equal(before, after)
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_net_is_deterministic_without_dropout_at_eval() -> None:
    model = TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=1)
    model.eval()
    batch = _toy_batch(4, 64, 12, 8)
    a = model(**batch)
    b = model(**batch)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_short_leading_valid_window_uses_packing() -> None:
    """A sample whose only valid rows are the last few must still produce finite output."""
    model = TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=1)
    model.eval()
    batch = _toy_batch(2, 128, 12, 8)
    batch["fast_mask"][0, :-2] = False  # only the last two rows valid
    batch["fast_mask"][1, :-5] = False  # only the last five rows valid
    soc, soh, tte = model(**batch)
    assert torch.isfinite(soc).all() and torch.isfinite(soh).all() and torch.isfinite(tte).all()


def test_net_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        TriStateLiteNet(fast_input_dim=0, slow_input_dim=8)
    with pytest.raises(ValueError, match="positive"):
        TriStateLiteNet(fast_input_dim=12, slow_input_dim=0)
    with pytest.raises(ValueError, match="layers"):
        TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        TriStateLiteNet(fast_input_dim=12, slow_input_dim=8, dropout=1.0)


def test_point_head_is_single_value_per_state() -> None:
    torch.manual_seed(2)
    head = PointHead(16)
    out = head(torch.randn(5, 16))
    assert out.shape == (5, 1)
    out.sum().backward()
    assert head.linear.weight.grad is not None


def test_mc_dropout_stochasticity_is_configurable() -> None:
    """train() mode should produce different outputs across calls (dropout active)."""
    model = TriStateLiteNet(
        fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=2, dropout=0.5
    )
    batch = _toy_batch(8, 64, 12, 8)
    model.train()
    a = model(**batch)
    b = model(**batch)
    assert any(not torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_tcn_encoder_outputs_anchor_representation() -> None:
    torch.manual_seed(9)
    model = TriStateLiteNet(
        fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=3,
        num_quantiles=3, head_mode="ordered", encoder_type="tcn",
    )
    batch = _toy_batch(6, 128, 12, 8)
    soc, soh, tte = model(**batch)
    assert soc.shape == (6, 3)
    assert torch.isfinite(soc).all() and torch.isfinite(tte).all()
    (soc.sum() + soh.sum() + tte.sum()).backward()
    assert model.gru.net[0][0].weight.grad is not None


def test_net_supports_eleven_ordered_quantiles() -> None:
    torch.manual_seed(3)
    model = TriStateLiteNet(
        fast_input_dim=12, slow_input_dim=8, hidden_dim=16, num_layers=1, num_quantiles=11
    )
    batch = _toy_batch(6, 128, 12, 8)
    soc, soh, tte = model(**batch)
    for out in (soc, soh, tte):
        assert out.shape == (6, 11)
        assert torch.all(out[:, :-1] <= out[:, 1:])  # fully ordered
        assert torch.isfinite(out).all()
    assert torch.all((soc >= 0) & (soc <= 1))
    assert torch.all((soh >= 0) & (soh <= 1))
    assert torch.all(tte >= 0)
