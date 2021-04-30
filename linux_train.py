import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

import time
import math
import subprocess

# python train.py --dataset csv --csv_train train_anno.csv  --csv_classes class_list.csv  --csv_val val_anno.csv --epochs 100
# python visualize.py --dataset csv  --csv_classes class_list.csv --csv_val val_anno.csv   --model csv_retinanet_188.pt

# %run train.py --dataset csv --csv_train train_annots.csv  --csv_classes class_list.csv  --csv_val val_annots.csv --epochs 20
# python visualize.py --dataset csv  --csv_classes class_list.csv  --csv_val val_annots.csv  --model csv_retinanet_7.pt

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent + .00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def main(args=None):


    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)
    SCALE=1.4 ############################## dikkat ####################
    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose(
                                       [Normalizer(), Resizer(SCALE=SCALE)]))  # Augmenter çıkartıldı. Augmenter(),

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer(SCALE=SCALE)]))



    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # ENABLE cuDNN AUTOTUNER
    torch.backends.cudnn.benchmark = True

    # num worker =4 and pin_memory = True
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, collate_fn=collater, batch_sampler=sampler,num_workers=4,pin_memory=True)

    if dataset_val is not None:
        # num worker =4 and pin_memory = True
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, collate_fn=collater, batch_sampler=sampler_val,num_workers=4,pin_memory=True)

    # # Create the model
    # if parser.depth == 18:
    #     retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    # elif parser.depth == 34:
    #     retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    # elif parser.depth == 50:
    #     retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    # elif parser.depth == 101:
    #     retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    # elif parser.depth == 152:
    #     retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet = model.RetinaNet(num_classes=dataset_train.num_classes())

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4, weight_decay=0.0000005)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.resnet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    best_mAP = -1
    total = {"train_loss": [], "val_loss": [], "train_mAP": [], "val_mAP": []}
    start_time = time.time()

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.resnet.freeze_bn()

        epoch_loss = []
        print("Cuda Control (1)", "------------------ epoch_num : ",epoch_num," ------------------")
        subprocess.run([r"nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                        "--format=csv"])

        for iter_num, data in enumerate(dataloader_train):

            try:
                optimizer.zero_grad()
                # zero_grad
                #for param in retinanet.parameters():
                #    param.grad = None

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet(
                        [data['img'].cuda().float(), torch.Tensor(data['annot']).cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # print(
                #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print("Cuda Control (2)")
        subprocess.run([r"nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                        "--format=csv"])

        val_epoch_loss = []
        retinanet.eval()
        retinanet.training = True
        with torch.no_grad():

            for iter_num, data in enumerate(dataloader_val):
                try:

                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet(
                            [data['img'].cuda().float(), torch.Tensor(data['annot']).cuda()])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss
                    if bool(loss == 0):
                        continue
                    val_epoch_loss.append(float(loss))
                except Exception as e:
                    print(e)
                    continue

        print("Cuda Control (3)")
        subprocess.run([r"nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                        "--format=csv"])

        retinanet.eval()
        # ------------------------   mAP  -----------------
        print('Evaluating dataset')
        print("val mAP")
        val_mAP = csv_eval.evaluate(dataset_val, retinanet)[0][0]  # ==> label==0 ==nuclei
        print("train mAP")
        train_mAP = csv_eval.evaluate(dataset_train, retinanet)[0][0]

        print("Cuda Control (4)")
        subprocess.run([r"nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                        "--format=csv"])

        # ------------------------   train loss  -----------------
        print(
            '(Train) {} ([{}/{}] % {})  Epoch: {} |  Running loss (classification_loss + regression_loss): {:1.5f}  | Train mAP : {}'.format(
                timeSince(start_time, (epoch_num + 1) / parser.epochs), epoch_num + 1, parser.epochs,
                                                                        (epoch_num + 1) / parser.epochs * 100,
                epoch_num, np.mean(epoch_loss), train_mAP))

        # ------------------------   val loss  -----------------
        print(
            '(Validation) Epoch: {} | Val mAP: {} | Running loss (classification_loss + regression_loss):{:1.5f}'.format(
                epoch_num, val_mAP, np.mean(val_epoch_loss)))

        # ------------------------   graph  -----------------

        total["train_loss"].append(np.mean(epoch_loss))
        total["val_loss"].append(np.mean(val_epoch_loss))
        total["train_mAP"].append(train_mAP)
        total["val_mAP"].append(val_mAP)

        retinanet.eval()
        retinanet.module.create_trace_file()
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

        # ------------------------   create tracefile  -----------------
        if val_mAP > best_mAP:

            best_mAP = val_mAP
            print("  **")

            retinanet.eval()
            retinanet.module.create_trace_file()
            torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

        else:

            print("")

        scheduler.step(np.mean(val_epoch_loss))  # scheduler.step(np.mean(epoch_loss))

    ############################# plot ###################################
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # The data
    x = [i for i in range(1, len(total["train_loss"]) + 1)]
    train_loss = total["train_loss"]
    val_loss = total["val_loss"]
    val_mAP = total["val_mAP"]
    train_mAP = total["train_mAP"]

    # Labels to use in the legend for each line
    line_labels = ["train_loss", "val_loss", "train_mAP", "val_mAP"]

    # Create the sub-plots, assigning a different color for each line.
    # Also store the line objects created
    l1 = ax1.plot(x, train_loss, color="red")[0]
    l2 = ax1.plot(x, val_loss, color="blue")[0]
    l3 = ax2.plot(x, train_mAP, color="red")[0]
    l4 = ax2.plot(x, val_mAP, color="blue")[0]

    # Create the legend
    fig.legend([l1, l2, l3, l4],  # The line objects
               labels=line_labels,  # The labels for each line
               borderaxespad=0.1,  # Small spacing around legend box
               title="Legend Title"  # Title for the legend
               )
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.85)
    fig.savefig('result.png')

    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
