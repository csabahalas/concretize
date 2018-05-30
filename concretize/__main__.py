import argparse

from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

import model as mod

def run_example(args, device):
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

def print_eval(model, criterion, loader, device, name="dev"):
    model.eval()
    total_accuracy = 0
    total_loss = 0
    total_count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores = model(inputs).detach()
        loss = criterion(scores, labels).detach()
        total_accuracy += (torch.max(scores, 1)[1] == labels).float().sum()
        total_loss += loss.item()
        total_count += scores.size(0)
    print(f"{name} accuracy: {total_accuracy / total_count:.4} loss: {total_loss / total_count:.4}")

def run_mnist(args, device):
    model = mod.ConvModel().to(device)
    train_set = MNIST(".", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = MNIST(".", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=100)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        print(f"Epoch #{epoch + 1}, beta: {model.beta:.3}")
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            if idx % args.dev_every == 0:
                print_eval(model, criterion, test_loader, device, name="test")
        model.beta *= 0.7
        torch.save(model.state_dict(), "mnist.pt")
    print_eval(model, criterion, test_loader, device, name="test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1E-3, type=float)
    parser.add_argument("--dev_every", default=5, type=int)
    parser.add_argument("--device", default=0, type=int, help="-1 for CPU, all else for GPU id")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--momentum", default=0, type=float)
    args = parser.parse_args()
    device = f"{'cuda' if args.device >= 0 else 'cpu'}:{args.device}"
    device = torch.device(device)
    run_mnist(args, device)

if __name__ == "__main__":
    main()
