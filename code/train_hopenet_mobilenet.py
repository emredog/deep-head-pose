import sys
import os
import argparse
import time
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

import datasets
import hopenet
import utils


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Head pose estimation using the Hopenet network."
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        help="Maximum number of training epochs.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=16, type=int
    )
    parser.add_argument(
        "--lr", dest="lr", help="Base learning rate.", default=0.001, type=float
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="Dataset type.",
        default="Pose_300W_LP",
        type=str,
    )
    parser.add_argument(
        "--val_dataset",
        dest="val_dataset",
        help="Dataset type.",
        default="AFLW2000",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Directory path for data.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        help="Directory path for snapshots and loss stats.",
        default="output/snapshots/",
        type=str,
    )
    parser.add_argument(
        "--filename_list",
        dest="filename_list",
        help="Path to text file containing relative paths for every example.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--val_filename_list",
        dest="val_filename_list",
        help="Path to text file containing relative paths for every example.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_string",
        dest="output_string",
        help="String appended to output snapshots.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        help="Regression loss coefficient.",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        help="Path of model snapshot.",
        default="",
        type=str,
    )

    args = parser.parse_args()
    return args


# FIXME use these functions for freezing layers
def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if "bn" in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if "bn" in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = args.num_epochs
    batch_size = args.batch_size

    if not os.path.exists("output/snapshots"):
        os.makedirs("output/snapshots")

    # ResNet50 structure
    model = hopenet.Hopenet_mobilenet(num_bins=66, pretrained=True)

    if not args.snapshot == "":
        print("Loading weights from ", args.snapshot)
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print("Loading data.")

    transformations = transforms.Compose(
        [
            transforms.Resize(240),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dataset == "Pose_300W_LP":
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "Pose_300W_LP_random_ds":
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "Synhead":
        pose_dataset = datasets.Synhead(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "AFLW2000":
        pose_dataset = datasets.AFLW2000(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "BIWI":
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == "AFLW":
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == "AFLW_aug":
        pose_dataset = datasets.AFLW_aug(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "AFW":
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print("Error: not a valid dataset name")
        sys.exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # FIXME
    assert args.val_dataset == "AFLW2000"
    val_dataset = datasets.AFLW2000(
        args.data_dir, args.val_filename_list, transformations
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size // 2, shuffle=False, num_workers=2
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).to(device)
    idx_tensor = [idx for idx in range(66)]
    # idx_tensor = Variable(torch.FloatTensor(idx_tensor))
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    optimizer = torch.optim.Adam(
        [
            # {"params": get_ignored_params(model), "lr": 0},
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": get_fc_params(model), "lr": args.lr * 5},
        ],
        lr=args.lr,
    )

    print("Ready to train network.")
    training_stats = {
        "loss_yaw": [],
        "loss_pitch": [],
        "loss_roll": [],
        "val_yaw_error": [],
        "val_pitch_error": [],
        "val_roll_error": [],
    }
    for epoch in range(num_epochs):
        start = time.time()
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = images.to(device)

            # Binned labels
            label_yaw = labels[:, 0].to(device)
            label_pitch = labels[:, 1].to(device)
            label_roll = labels[:, 2].to(device)

            # Continuous labels
            label_yaw_cont = cont_labels[:, 0].to(device)
            label_pitch_cont = cont_labels[:, 1].to(device)
            label_roll_cont = cont_labels[:, 2].to(device)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            training_stats["loss_yaw"].append(loss_yaw)
            training_stats["loss_pitch"].append(loss_pitch)
            training_stats["loss_roll"].append(loss_roll)

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1).to(device) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Iter [{}/{}] Losses: Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(pose_dataset) // batch_size,
                        loss_yaw.item(),
                        loss_pitch.item(),
                        loss_roll.item(),
                    )
                )

        elapsed_time = time.time() - start

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print(
                "Epoch completed in {:.1f} seconds. Taking snapshot...".format(
                    elapsed_time
                )
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.out_dir,
                    args.output_string + "_epoch_" + str(epoch + 1) + ".pkl",
                ),
            )

            # VALIDATE on "AFLW2000"
            model.eval()
            total = 0

            idx_tensor = [idx for idx in range(66)]
            idx_tensor = torch.FloatTensor(idx_tensor).to(device)

            yaw_error = 0.0
            pitch_error = 0.0
            roll_error = 0.0

            for i, (images, labels, cont_labels, name) in enumerate(val_loader):
                images = images.to(device)
                total += cont_labels.size(0)

                label_yaw = cont_labels[:, 0].float()
                label_pitch = cont_labels[:, 1].float()
                label_roll = cont_labels[:, 2].float()

                yaw, pitch, roll = model(images)

                # Binned predictions
                _, yaw_bpred = torch.max(yaw.data, 1)
                _, pitch_bpred = torch.max(pitch.data, 1)
                _, roll_bpred = torch.max(roll.data, 1)

                # Continuous predictions
                yaw_predicted = utils.softmax_temperature(yaw.data, 1)
                pitch_predicted = utils.softmax_temperature(pitch.data, 1)
                roll_predicted = utils.softmax_temperature(roll.data, 1)

                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
                pitch_predicted = (
                    torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
                )
                roll_predicted = (
                    torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
                )

                # Mean absolute error
                yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
                pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
                roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

            yaw_error = yaw_error / total
            pitch_error = pitch_error / total
            roll_error = roll_error / total
            training_stats["val_yaw_error"].append(yaw_error)
            training_stats["val_pitch_error"].append(pitch_error)
            training_stats["val_roll_error"].append(roll_error)
            print(
                "Validation error in degrees of the model on the "
                + str(total)
                + " test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f"
                % (yaw_error, pitch_error, roll_error)
            )
            model.train()  # back to training mode

            with open(
                os.path.join(args.out_dir, "losses" + args.output_string + ".pkl"), "wb"
            ) as handle:
                pkl.dump(training_stats, handle)
