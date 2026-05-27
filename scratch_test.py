import torch

log_weights = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)
opt = torch.optim.Adam([log_weights], lr=0.1)

def biased_val_fn(alpha, beta, gamma):
    # Changed 0.1 to 0.4 so it's not clamped to zero
    sim_t = torch.tensor([0.9, 0.4], dtype=torch.float32)
    acc_t = torch.tensor([0.5, 0.5], dtype=torch.float32)
    anom_t = torch.tensor([0.1, 0.9], dtype=torch.float32)
    raw = torch.clamp(alpha * sim_t + beta * acc_t - gamma * anom_t, 0.0, 1.0)
    w = raw / (raw.sum() + 1e-8)
    losses = torch.tensor([0.3, 1.2], dtype=torch.float32)
    return (w * losses).sum()

for i in range(5):
    opt.zero_grad()
    w = torch.softmax(log_weights, dim=0)
    loss = biased_val_fn(w[0], w[1], w[2])
    loss.backward()
    print('grad:', log_weights.grad)
    opt.step()
    print('w:', torch.softmax(log_weights, dim=0))
