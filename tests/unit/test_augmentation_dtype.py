import pytest
import torch

from sepsis_most.training.augment import jitter


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_jitter_preserves_batch_dtype_and_inputs(dtype):
    torch.manual_seed(9)
    x = torch.ones(2, 3, 8, dtype=dtype)
    s = torch.ones(2, 4, dtype=dtype)

    augmented_x, augmented_s = jitter(x, s, sigma=0.1)

    assert augmented_x.dtype == dtype
    assert augmented_s.dtype == dtype
    assert torch.isfinite(augmented_x).all() and torch.isfinite(augmented_s).all()
    torch.testing.assert_close(x, torch.ones_like(x))
    torch.testing.assert_close(s, torch.ones_like(s))
