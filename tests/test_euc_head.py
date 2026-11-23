import torch

from src.models.euc_head import EvidentialHead


def _batch():
    batch, steps, dim, classes = 3, 4, 16, 5
    f = torch.randn(batch, steps, dim, requires_grad=True)
    g = [torch.randn(batch, steps, dim, requires_grad=True) for _ in range(3)]
    y = torch.randint(0, classes, (batch, steps))
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32
    )
    return f, g, y, mask


def test_unimodal_anchor_trains_all_modal_heads():
    head = EvidentialHead(
        d_f=16,
        d_modal=16,
        num_classes=5,
        lambda_ce=1.0,
        lambda_uni=0.5,
    )
    f, g, y, mask = _batch()
    out, losses = head(f, *g, y, epoch=0, mask_t=mask)

    losses["L_head"].backward()

    assert torch.isfinite(losses["L_uni"])
    for modal_head in (head.modal_head_v, head.modal_head_a, head.modal_head_l):
        assert modal_head.weight.grad is not None
        assert modal_head.weight.grad.abs().sum() > 0


def test_calibration_uses_correctness_hard_target():
    """L_cal 目标是 1[ŷ≠y] 硬标签，且该目标对三个模态辅助头 stop-gradient。"""
    head = EvidentialHead(
        d_f=16,
        d_modal=16,
        num_classes=5,
        lambda_cal=0.5,
    )
    f, g, y, mask = _batch()
    out, losses = head(f, *g, y, epoch=3, mask_t=mask)

    losses["L_cal"].backward()

    # L_cal 只更新证据不确定性分支（evidence_net），不更新模态辅助头
    assert head.evidence_net[0].weight.grad is not None
    assert head.modal_head_v.weight.grad is None
    assert head.modal_head_a.weight.grad is None
    assert head.modal_head_l.weight.grad is None


def test_sequence_loss_uses_masked_single_prior_aggregation_and_backpropagates():
    head = EvidentialHead(
        d_f=16,
        d_modal=16,
        num_classes=5,
        lambda_seq=0.5,
    )
    f, g, y, mask = _batch()
    seq_y = torch.tensor([1, 2, 3])
    out, losses = head(f, *g, y, epoch=3, mask_t=mask, seq_y=seq_y)

    manual = 1.0 + ((out["alpha"] - 1.0) * mask.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(out["alpha_seq"], manual)
    assert out["p_seq"].shape == (3, 5)
    assert out["y_seq_pred"].shape == (3,)
    assert torch.allclose(
        losses["L_head"], losses["L_unit"] + 0.5 * losses["L_seq"]
    )

    losses["L_seq"].backward()
    assert head.evidence_net[0].weight.grad is not None
    assert head.evidence_net[0].weight.grad.abs().sum() > 0
