"""TRITON_INTERPRET is read when triton is IMPORTED, and other test modules load it (through torch) before
test_fused_nn would: set here, before any module is collected, so the fused kernels run in Triton's
interpreter wherever there is no card."""
import os

import torch

if not torch.cuda.is_available():
    os.environ['TRITON_INTERPRET'] = '1'
