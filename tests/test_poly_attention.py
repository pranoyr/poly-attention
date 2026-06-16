import pytest
import torch
from poly_attention import PolyAttention

@pytest.mark.parametrize('causal', (False, True))
def test_poly_attention(causal):
    attn = PolyAttention(dim = 128, heads = 4, dim_head = 32, causal = causal)
    x = torch.randn(2, 32, 128)

    out = attn(x)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

def test_poly_attention_mask():
    attn = PolyAttention(dim = 128, heads = 4, dim_head = 32)
    x = torch.randn(2, 32, 128)

    mask = torch.ones(2, 32).bool()
    mask[:, 16:] = False

    out = attn(x, mask = mask)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()
