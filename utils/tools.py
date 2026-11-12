import math


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))
