import torch
import torch.nn, torch.optim, torch.cuda
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader

from dataloader import mnist_loader as ml
from models.cnnnet import CnnNet


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train():
    os.makedirs('./output', exist_ok=True)
    ml.image_list(args.datapath, 'output/total.txt')
    ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

    train_data = ml.MyDataset(txt='output/train.txt', transform=transforms.ToTensor())
    val_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    '''
    # 调用内置经典模型方式
    resnet18 = models.resnet18()
    fc_features = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(fc_features, 10)
    model = resnet18
    '''
    model = CnnNet()
    # model.load_state_dict(torch.load('./output/params_10.pth'))

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        i = 0
        for batch_x, batch_y in train_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            i += 1
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch+1, args.epochs, i, len(train_data)/args.batch_size+1,
                     loss.item(), train_correct.item()/len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (len(train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (len(val_data)), eval_acc / (len(val_data))))

        #torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
        torch.save(model.state_dict(), 'output/params_' + str(epoch+1) + '.pth')


if __name__ == '__main__':
    train()
