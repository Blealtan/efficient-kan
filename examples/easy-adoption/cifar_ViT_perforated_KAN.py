import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

import efficient_kan
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

def kan_from_linear(linear_module):
    """
    Create a KANLinear module from a standard torch.nn.Linear module.
    
    Args:
        linear_module: A torch.nn.Linear module to extract dimensions from
    Returns:
        A KANLinear module with the same input/output dimensions as linear_module
    """    
    
    # Adjustable settings here
    grid_size=5
    spline_order=3
    scale_noise=0.1
    scale_base=1.0
    scale_spline=1.0
    enable_standalone_scale_spline=True
    base_activation=torch.nn.SiLU
    grid_eps=0.02
    grid_range=[-1, 1]
    return efficient_kan.KANLinear(
        in_features=linear_module.in_features,
        out_features=linear_module.out_features,
        grid_size=grid_size,
        spline_order=spline_order,
        scale_noise=scale_noise,
        scale_base=scale_base,
        scale_spline=scale_spline,
        enable_standalone_scale_spline=enable_standalone_scale_spline,
        base_activation=base_activation,
        grid_eps=grid_eps,
        grid_range=grid_range,
    )


def setupKANonfiguration():
    # Instruct PAI to replace all nn.Linears with a call to kan_from_linear with that nn.Linear as the argument
    GPA.pc.append_modules_to_replace([nn.Linear])
    GPA.pc.append_replacement_modules([kan_from_linear])
    # This line tells the system to not convert the new KANLinears to be dendrite modules (or the other conv2d modules)
    GPA.pc.append_modules_to_track([efficient_kan.KANLinear, nn.Conv2d])
    # These lines tell the system we are not doing dendritic optimization at all and shut off warnings related to dendrites
    GPA.pc.set_max_dendrites(0)
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_checked_skipped_modules(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)


def get_vit_model(num_classes=10):
    """
    Get a Vision Transformer model for CIFAR-10.
    Using a very small ViT variant with minimal layers.
    """
    # Load a very small ViT model - reduced depth and dimensions for memory efficiency
    # embed_dim=192, depth=4 (only 4 transformer blocks instead of 12)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes, 
                             img_size=64, embed_dim=192, depth=4)
    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 ViT Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.05, metavar='W',
                        help='weight decay (default: 0.05)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # CIFAR-10 normalization values, resized to 64x64 for smaller ViT
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform_train)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    model = get_vit_model(num_classes=10).to(device)
    print('Original Model (before KAN conversion):')
    print(model)
    print('\n' + '='*80 + '\n')
    
    # Count nn.Linear layers before conversion
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f'Number of nn.Linear layers: {linear_count}')
    print('Converting all nn.Linear layers to KANLinear...\n')
    
    model = UPA.initialize_pai(model).to(device)
    print('KAN Model (after conversion):')
    print(model)
    
    # Count KANLinear layers after conversion
    kan_count = sum(1 for m in model.modules() if isinstance(m, efficient_kan.KANLinear))
    print(f'\nNumber of KANLinear layers: {kan_count}')
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing scheduler works well with ViTs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_vit_kan.pt")


if __name__ == '__main__':
    setupKANonfiguration()
    main()
