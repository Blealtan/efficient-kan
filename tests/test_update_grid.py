import torch

from efficient_kan import KANLinear


def test_update_grid_keeps_output():
    torch.manual_seed(0)
    layer = KANLinear(3, 2)
    with torch.no_grad():
        # the default init leaves the spline part tiny, so make it visible
        layer.spline_weight.normal_()

    x = torch.rand(512, 3) * 2 - 1
    with torch.no_grad():
        before = layer(x)
        layer.update_grid(x)
        after = layer(x)

    # refitting on the new grid is not exact, so allow a small change
    change = ((after - before).norm() / before.norm()).item()
    assert change < 0.1, f"output changed by {change:.2f} after update_grid"
