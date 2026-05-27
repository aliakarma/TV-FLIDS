"""
data/preprocessing/mnist_fl_pipeline.py
MNIST FL benchmark for task-agnosticism demonstration.
Reference: Guide §19
"""

import numpy as np

try:
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


def load_mnist_fl(num_clients: int = 20, alpha: float = 0.5,
                   data_root: str = "data/raw/mnist", seed: int = 42):
    """
    Load MNIST and partition into non-IID client datasets.

    Args:
        num_clients: Number of FL clients.
        alpha: Dirichlet concentration (0.5=realistic, 0.1=extreme).
        data_root: Directory to store/load MNIST.
        seed: Random seed.

    Returns:
        client_data: List of (X, y) tuples per client.
        X_test: Global test features.
        y_test: Global test labels.
        class_weights: np.ndarray of per-class weights.
    """
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for MNIST. pip install torchvision")

    import os
    os.makedirs(data_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy().astype(np.int64)
    X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy().astype(np.int64)

    # Reuse NonIIDPartitioner — same strategy as NSL-KDD
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from data.partitioning import NonIIDPartitioner

    partitioner = NonIIDPartitioner(alpha=alpha)
    client_data = partitioner.partition(X_train, y_train, num_clients, seed=seed)

    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced",
                                    classes=np.arange(10), y=y_train)

    print(f"[MNIST-FL] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Clients: {num_clients}, α={alpha}")
    return client_data, X_test, y_test, weights
