import logging
import time
# from time import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import AverageMeter, save
import datetime
import os

def train_cnn(
    source_cnn, train_loader, test_loader, criterion, optimizer,
    args=None
):

    now = datetime.datetime.today()
    if args.test and args.arch == 'FFT_CNNEx':
        log_path = os.path.join(args.logdir,format(now,'t_fft_%m%d_%H:%M-%S.log'))
    elif args.test and args.arch == 'CNNEx':
        log_path = os.path.join(args.logdir,format(now,'t_cnn_%m%d_%H:%M-%S.log'))
    elif args.arch == 'FFT_CNNEx':
        log_path = os.path.join(args.logdir,format(now,'fft_%m%d_%H:%M-%S.log'))
    elif args.arch == 'CNNEx':
        log_path = os.path.join(args.logdir,format(now,'cnn_%m%d_%H:%M-%S.log'))
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path,level=logging.INFO)


    pre_info = '{} | {}'.format(args.arch, args.dataset)
    logger.info(pre_info)
    pre_info = 'in_size : {} | epochs : {} | batch : {}'.format(args.size, args.epochs, args.batch_size)
    logger.info(pre_info)
    if args.kernel_l1:
        kernel_log = 'kernel_l1 : {} | '.format(args.kernel_l1)
    if args.kernel_l2:
        kernel_log += 'kernel_l2 : {} | '.format(args.kernel_l2)
    if args.kernel_l3:
        kernel_log += 'kernel_l3 : {} '.format(args.kernel_l3)
    if args.kernel_l4:
        kernel_log += 'kernel_l4 : {} | '.format(args.kernel_l4)
    if args.kernel_l5:
        kernel_log += 'kernel_l5 : {} '.format(args.kernel_l5)
    logger.info(kernel_log)
    if args.o_channels_l1:
        o_channels_log = 'o_channels_l1 : {} | '.format(args.o_channels_l1)
    if args.o_channels_l2:
        o_channels_log += 'o_channels_l2 : {} | '.format(args.o_channels_l2)
    if args.o_channels_l3:
        o_channels_log += 'o_channels_l3 : {} '.format(args.o_channels_l3)
    if args.o_channels_l4:
        o_channels_log += 'o_channels_l4 : {} | '.format(args.o_channels_l4)
    if args.o_channels_l5:
        o_channels_log += 'o_channels_l5 : {} '.format(args.o_channels_l5)
    logger.info(o_channels_log)
    if args.padding_l1:
        padding_log = 'padding_l1 : {} | '.format(args.padding_l1)
    if args.padding_l2:
        padding_log += 'padding_l2 : {} | '.format(args.padding_l2)
    if args.padding_l3:
        padding_log += 'padding_l3 : {} '.format(args.padding_l3)
    if args.padding_l4:
        padding_log += 'padding_l4 : {} | '.format(args.padding_l4)
    if args.padding_l5:
        padding_log += 'padding_l5 : {} '.format(args.padding_l5)
    if args.padding_l1 or args.padding_l2 or args.padding_l3 or args.padding_l4 or args.padding_l5:
        logger.info(padding_log)
    learn_time = 'learn_num_train : {} | '.format(args.learnnum_t)
    learn_time += 'learn_num_validate : {} '.format(args.learnnum_v)
    logger.info(learn_time)
    # if args.test:
    #     logger.info(learn_time)

    if args.comment:
        logger.info(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        comment_log = 'message : {} '.format(args.comment)
        logger.info(comment_log)

    logger.info(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')

    total_time = 0.0
    best_score = None
    time = []
    for epoch_i in range(1, 1 + args.epochs):
        # start_time = time()
        time, Ave_Time = train(
            source_cnn, train_loader, criterion, optimizer, args=args)
        Timelog = 'Ave_Time : {}'.format(Ave_Time)
        logger.info(Timelog)
        logger.info(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    for result in time:
        logger.info(result)
        # validation = validate(
        #     source_cnn, test_loader, criterion, args=args)
        # log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
        # log += '| Train/Loss {:.3f} Acc {:.3f} '.format(
        #     training['loss'], training['acc'])
        # log += '| Val/Loss {:.3f} Acc {:.3f} '.format(
        #     validation['loss'], validation['acc'])
        # total_time += time() - start_time
        # log += 'Time {:.2f}s'.format(time() - start_time)
        # if epoch_i == args.epochs:
        #     average_time = 'Average_Time {:.2f}s'.format(total_time / args.epochs)
        # print(log)
        # logger.info(log)

        # save
    #     is_best = (best_score is None or validation['acc'] > best_score)
    #     best_score = validation['acc'] if is_best else best_score
    #     state_dict = {
    #         'model': source_cnn.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'epoch': epoch_i,
    #         'val/acc': best_score,
    #     }
    #     save(args.logdir,'pretrain',state_dict, is_best)
    # log = 'Best validation accuracy:{:.3f}%'.format(best_score)
    # logger.info(log)
    # logger.info(average_time)
    # print(log)
    # print(average_time)
    return source_cnn

def step(model, data, target, criterion, args):
    data, target = data.to(args.device), target.to(args.device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        # output, result_time = model(data)
        output = model(data)
    torch.cuda.synchronize()
    result_time = (time.time() - start) * 1000

    # loss = criterion(output, target)
    return output, result_time
    # return output, loss


def train(model, dataloader, criterion, optimizer, args=None):
    model.train()
    losses = AverageMeter()
    targets, probas, info = [], [], []
    time, total = 0.0, 0.0
    count = 0
    for i, (data, target) in enumerate(dataloader):

        #print(data.shape,target.shape)
        #sys.exit(0)

        bs = target.size(0)
        output, time = step(model, data, target, criterion, args)
        # output, loss = step(model, data, target, criterion, args)
        print(time)
        if i >= 10 :
            info = np.append(info, time)
            total += time
            count += 1
        # output = torch.softmax(output, dim=1)  # NOTE
        # losses.update(loss.item(), bs)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # targets.extend(target.cpu().detach().numpy().tolist())
        # probas.extend(output.cpu().detach().numpy().tolist())
        # Warning
        if i == args.learnnum_t - 1:
            break
    Ave_Time = total/count
    print('Ave_Time : ' + str(Ave_Time))
    print('count : ' + str(count))
    return info, Ave_Time
    # probas = np.asarray(probas)
    # preds = np.argmax(probas, axis=1)
    # acc = accuracy_score(targets, preds)
    # return {
    #     'loss': losses.avg, 'acc': acc,
    # }


def validate(model, dataloader, criterion, args=None):
    model.eval()
    losses = AverageMeter()
    targets, probas = [], []
    time, total = 0.0, 0.0

    with torch.no_grad():
        for iter_i, (data, target) in enumerate(dataloader):
            bs = target.size(0)
            output = step(model, data, target, criterion, args)
            # output, loss = step(model, data, target, criterion, args)
            total += time
            output = torch.softmax(output, dim=1)  # NOTE: check
            # losses.update(loss.item(), bs)
            # targets.extend(target.cpu().numpy().tolist()) #torch.no_grad():で勾配を考慮しない手順によってnumpyが使えている説
            # probas.extend(output.cpu().numpy().tolist())
            
            # Warning
            if iter_i == args.learnnum_v - 1:
                break
    # print(total/args.learnnum_t)
    return
    # probas = np.asarray(probas)
    # preds = np.argmax(probas, axis=1)
    # acc = accuracy_score(targets, preds)
    # return {
    #     'loss': losses.avg, 'acc': acc*100,
    # }
