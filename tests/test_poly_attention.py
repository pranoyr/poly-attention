import pytest
import torch
from poly_attention import PolyAttention

param = pytest.mark.parametrize

@param('causal', (False, True))
@param('kv_heads', (1, 2, 4))
def test_poly_attention(causal, kv_heads):
    attn = PolyAttention(dim = 128, heads = 4, kv_heads = kv_heads, dim_head = 32, causal = causal)
    x = torch.randn(2, 32, 128)

    out = attn(x)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

@param('kv_heads', (1, 2, 4))
def test_poly_attention_mask(kv_heads):
    attn = PolyAttention(dim = 128, heads = 4, kv_heads = kv_heads, dim_head = 32)
    x = torch.randn(2, 32, 128)

    mask = torch.ones(2, 32).bool()
    mask[:, 16:] = False

    out = attn(x, mask = mask)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

def test_invalid_kv_heads():
    with pytest.raises(AssertionError):
        # heads (4) is not divisible by kv_heads (3)
        PolyAttention(dim = 128, heads = 4, kv_heads = 3, dim_head = 32)
