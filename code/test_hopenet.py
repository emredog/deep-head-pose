import sys
import os
import argparse
import time

import cv2

import torch
from torchvision import transforms
import torchvision

import datasets
import hopenet
import utils

backbones = ["resnet", "mobilenet", "shufflenet", "squeezenet"]


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Head pose estimation using the Hopenet network."
    )

    parser.add_argument(
        "--backbone",
        dest="backbone",
        help="Model backbone to use. Default is Resnet50. Avaible values: {}".format(
            backbones
        ),
        default="resnet",
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
        "--filename_list",
        dest="filename_list",
        help="Path to text file containing relative paths for every example.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--mobilenet_width",
        dest="mobilenet_width",
        help="Width coef of the MobileNet.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--shufflenet_mult",
        dest="shufflenet_mult",
        help="ShuffleNet capacity multiplier",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        help="Name of model snapshot.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument(
        "--save_viz",
        dest="save_viz",
        help="Save images with pose cube.",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--dataset", dest="dataset", help="Dataset type.", default="AFLW2000", type=str
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    snapshot_path = args.snapshot

    if args.backbone not in backbones:
        raise ValueError(
            "{} is not recognized as a backbone. Please select one of the following: {}".format(
                args.backbone, backbones
            )
        )

    if args.backbone == "resnet":  # ResNet50 structure
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    elif args.backbone == "mobilenet":  # mobilenet backbone
        model = hopenet.Hopenet_mobilenet(
            num_bins=66, width_mult=args.mobilenet_width, pretrained=False
        )
    elif args.backbone == "shufflenet":  # shufflenet backbone
        model = hopenet.Hopenet_shufflenet(
            num_bins=66, shufflenet_mult=args.shufflenet_mult, pretrained=False
        )
    elif args.backbone == "squeezenet":  # squeezenet backbone
        model = hopenet.Hopenet_shufflenet(num_bins=66, pretrained=False)

    print("Loading snapshot.")
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device(device))
    model.load_state_dict(saved_state_dict)

    print("Loading data.")

    transformations = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
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
    elif args.dataset == "AFLW2000":
        pose_dataset = datasets.AFLW2000(
            args.data_dir, args.filename_list, transformations
        )
    elif args.dataset == "AFLW2000_ds":
        pose_dataset = datasets.AFLW2000_ds(
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
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset, batch_size=args.batch_size, num_workers=2
    )

    model.to(device)

    print("Ready to test network.")

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    yaw_error = 0.0
    pitch_error = 0.0
    roll_error = 0.0

    l1loss = torch.nn.L1Loss(size_average=False)
    inference_times = []

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        images = images.to(device)
        total += cont_labels.size(0)

        label_yaw = cont_labels[:, 0].float()
        label_pitch = cont_labels[:, 1].float()
        label_roll = cont_labels[:, 2].float()

        start = time.time()
        yaw, pitch, roll = model(images)
        elapsed = time.time() - start
        inference_times.append(elapsed)

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        # Mean absolute error
        yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
        pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
        roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            if args.dataset == "BIWI":
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + "_rgb.png"))
            else:
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + ".jpg"))
            if args.batch_size == 1:
                error_string = "y %.2f, p %.2f, r %.2f" % (
                    torch.sum(torch.abs(yaw_predicted - label_yaw)),
                    torch.sum(torch.abs(pitch_predicted - label_pitch)),
                    torch.sum(torch.abs(roll_predicted - label_roll)),
                )
                cv2.putText(
                    cv2_img,
                    error_string,
                    (30, cv2_img.shape[0] - 30),
                    fontFace=1,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                )
            # utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
            utils.draw_axis(
                cv2_img,
                yaw_predicted[0],
                pitch_predicted[0],
                roll_predicted[0],
                tdx=200,
                tdy=200,
                size=100,
            )
            cv2.imwrite(os.path.join("output/images", name + ".jpg"), cv2_img)

    print(
        "Test error in degrees of the model on the "
        + str(total)
        + " test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f"
        % (yaw_error / total, pitch_error / total, roll_error / total)
    )

    print(
        "Average inference time per image: {:.6f}ms".format(
            1000 * (sum(inference_times) / total)
        )
    )

