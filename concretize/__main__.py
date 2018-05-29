import argparse

import torch
import torch.optim as optim

import model as mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=1E-3, type=float)
    parser.add_argument("--dev_every", default=10, type=int)
    parser.add_argument("--device", default=0, type=int, help="-1 for CPU, all else for GPU id")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--momentum", default=0, type=float)
    args = parser.parse_args()
    device = f"{'cuda' if args.device >= 0 else 'cpu'}:{args.device}"
    device = torch.device(device)

    model = mod.MultiTuckerExampleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        model.train()
        loss = model.mse(args.batch_size)
        loss.backward()
        optimizer.step()
        if epoch % args.dev_every == 0:
            model.eval()
            loss = 0.5 * model.mse(1024)
            print(f"Epoch: {epoch} Loss: {loss.item():.5} Params: {[p.detach().cpu().sigmoid().numpy() for p in model.parameters()]}")

if __name__ == "__main__":
    main()
