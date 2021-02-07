'''
    Used to train models on CIFAR-100 and Tiny ImageNet
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import argparse
import time
from datetime import timedelta

from utils import calculate_acc, get_network, get_dataloader, init_params, count_parameters

parser = argparse.ArgumentParser(description='Training CNN models')

parser.add_argument('--network', '-n', required=True)
parser.add_argument('--epoch', '-e', type=int, default=90, help='Number of epochs')
parser.add_argument('--batch', '-b', type=int, default=128, help='The batch size')
parser.add_argument('--lr', '-l', type=float, default=0.01, help='Learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--weight-decay', '-d', type=float, default=0.0005, help='Weight decay for SGD optimizer')
parser.add_argument('--step-size', '-s', type=int, default=30, help='Step in learning rate scheduler')
parser.add_argument('--gamma', '-g', type=float, default=0.1, help='Gamma in learning rate scheduler')
parser.add_argument('--dataset', type=str, help='cifar100 or imagenet', default='cifar100')
parser.add_argument('--resume', type=str, help='resume training')
parser.add_argument('--save', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)

args = parser.parse_args()
print(args)

RESULT_FILE = 'results.txt'
LOG_FILE = 'logs/{}-{}-b{}-e{}.txt'.format(args.network, args.dataset, args.batch, args.epoch)

# Dict to keep the final result
stats = {
    'best_acc': 0.0,
    'best_epoch': 0
}

# Device
device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

# Dataloader
trainloader, testloader = get_dataloader(args.dataset, args.batch)

if args.dataset == 'cifar100' or args.dataset == 'tiny':
    VAL_LEN = 10000
elif args.dataset == 'imagenet':
    VAL_LEN = 150000

# Get network
net = get_network(args.network, args.dataset, device)

# Handle multi-gpu
if args.cuda and args.ngpu > 1:
    net = nn.DataParallel(net, list(range(args.ngpu)))

# Init parameters
init_params(net)

print('Training {} with {} parameters...'.format(args.network, count_parameters(net)))

net.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

if args.save and not args.resume:
    # Log basic hyper-params to log file
    with open(LOG_FILE, 'w') as f:
        f.write('Training model {}\n'.format(args.network))
        f.write('Hyper-parameters:\n')
        f.write('Epoch {}; Batch {}; LR {}; SGD Momentum {}; SGD Weight Decay {};\n'.format(str(args.epoch), str(args.batch), str(args.lr), str(args.momentum), str(args.weight_decay)))
        f.write('LR Scheduler Step {}; LR Scheduler Gamma {}; {};\n'.format(str(args.step_size), str(args.gamma), str(args.dataset)))
        f.write('Epoch,TrainLoss,ValAcc\n')

if args.resume is not None:
    checkpoint_path = args.resume
    state = torch.load(checkpoint_path)
    optimizer.load_state_dict(state['optimizer'])
    net.load_state_dict(state['net'])
    start_epoch = state['epoch']
    stats = state['stats'] if state['stats'] else { 'best_acc': 0.0, 'best_epoch': 0 }
else:
    checkpoint_path = 'trained_nets/{}-{}-b{}-e{}.tar'.format(args.network, args.dataset, args.batch, args.epoch)
    start_epoch = 0

# Train the model
start = time.time()
for epoch in range(start_epoch, args.epoch):  # loop over the dataset multiple times

    training_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    # Calculate validation accuracy
    net.eval()
    val_acc = calculate_acc(testloader, net, device)
    if val_acc > stats['best_acc']:
        stats['best_acc'] = val_acc
        stats['best_epoch'] = epoch + 1
        if args.save:
            # Save the checkpoint
            state = {
                'epoch': epoch, 
                'optimizer': optimizer.state_dict(),
                'net': net.state_dict(),
                'stats': stats
            }
            torch.save(state, checkpoint_path)

    # Switch back to training mode
    net.train()

    print('[Epoch: %d]  Train Loss: %.3f   Val Acc: %.3f%%' % ( epoch + 1, training_loss / len(trainloader), val_acc ))
    
    if args.save:
        with open(LOG_FILE, 'a+') as f:
            f.write('%d,%.3f,%.3f\n' % (epoch + 1, training_loss / len(trainloader), val_acc))

    # Step the scheduler after every epoch
    scheduler.step()

end = time.time()
print('Total time trained: {}'.format( str(timedelta(seconds=int(end - start)) ) ))

# Test the model
print('Test Accuracy of the {} on the {} test images: Epoch {}, {} % '.format(args.network, VAL_LEN, stats['best_epoch'], stats['best_acc']))
if args.save:
    with open(LOG_FILE, 'a+') as f:
        f.write('Total time trained: {}\n'.format( str(timedelta(seconds=int(end - start)) ) ))
        f.write('Test Accuracy of the {} on the {} test images: Epoch {}, {} %'.format(args.network, VAL_LEN, stats['best_epoch'], stats['best_acc']))

    with open(RESULT_FILE, 'a+') as f:
        f.write('**********************\n')
        f.write('Results of network {} on dataset {}:\n'.format(args.network, args.dataset))
        f.write('Accuracy: {}, Epoch: {}, Time: {}\n'.format(stats['best_acc'], stats['best_epoch'], str(timedelta(seconds=int(end - start)) ) ))