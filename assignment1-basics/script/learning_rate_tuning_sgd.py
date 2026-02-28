import torch

from src.optimizer import SGD

def run_training(lr=1):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    # lr = 1e1, 1e2 decay getting faster; lr = 1e3 getting diverge 
    print(f"lr: {lr}")
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimize
    print("---------")

if __name__ == '__main__':
    lrs = [1, 1e1, 1e2, 1e3]
    for lr in lrs:
        run_training(lr=lr)
