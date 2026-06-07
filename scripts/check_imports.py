import sys

try:
    import torch
    print('torch', torch.__version__)
except Exception as e:
    print('torch import error:', type(e).__name__, e)

try:
    import flwr
    print('flwr', flwr.__version__)
except Exception as e:
    print('flwr import error:', type(e).__name__, e)

try:
    import ray
    print('ray', ray.__version__)
except Exception as e:
    print('ray import error:', type(e).__name__, e)
