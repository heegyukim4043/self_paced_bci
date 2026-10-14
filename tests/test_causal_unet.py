import torch

from uncued_mi.causal_unet import CausalUNet1D


def test_shape_and_strict_causality():
    torch.manual_seed(2)
    model = CausalUNet1D(4, width=8).eval()
    x = torch.randn(2, 4, 251)
    with torch.no_grad():
        y = model(x)
        changed = x.clone(); changed[:, :, 180:] += 10
        y_changed = model(changed)
    assert y.shape == (2, 251)
    torch.testing.assert_close(y[:, :180], y_changed[:, :180])
