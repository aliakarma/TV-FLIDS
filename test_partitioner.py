from data.partitioning import NonIIDPartitioner
import numpy as np
X = np.random.randn(1000, 41).astype('float32')
y = np.random.randint(0, 5, 1000).astype('int64')
p = NonIIDPartitioner(0.5)
r1 = p.partition(X, y, 5, seed=42)
r2 = p.partition(X, y, 5, seed=42)
assert all(np.array_equal(r1[i][1], r2[i][1]) for i in range(5)), 'Non-deterministic!'
print('PASS: Partitioner is deterministic')
