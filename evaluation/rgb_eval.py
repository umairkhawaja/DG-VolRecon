import argparse
import os
import multiprocessing as mp
import numpy as np
from skimage.metrics import structural_similarity
from skimage.measure import compare_ssim
import torch
import lpips
import cv2
import pandas as pd


def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(image1, image2):
    rgb = image1.astype(np.float64) / 255.0
    gt = image2.astype(np.float64) / 255.0
    ssim_value = structural_similarity(rgb, gt, multichannel=True)
    return ssim_value


def calculate_lpips(image1, image2):
    # LPIPS requires input images as torch tensors
    lpips_model = lpips.LPIPS(net="alex")  # Using AlexNet
    image1_tensor = (
        torch.from_numpy(image1.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    image2_tensor = (
        torch.from_numpy(image2.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    lpips_value = lpips_model(image1_tensor, image2_tensor)
    return lpips_value.item()


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_view", type=int, default=3)
    parser.add_argument("--eval_output_dir", type=str, default="eval_outputs/")
    parser.add_argument("--dataset_dir", type=str, default="/home/dataset/DTU_TEST/")

    args = parser.parse_args()

    outdir = os.path.join(args.eval_output_dir, args.exp_name)

    all_psnr = []
    all_ssim = []
    all_lpips = []
    scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    view_list = [23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25]
    view_list = view_list[: args.num_view]

    for scan in scans:
        scan_folder = f"scan{scan}"
        rgb_dir = os.path.join(outdir, "rgb", scan_folder)
        depth_dir = os.path.join(outdir, "depth", scan_folder)
        gt_rgb_dir = os.path.join(args.dataset_dir, scan_folder, "image")

        scan_psnr = []
        scan_ssim = []
        scan_lpips = []
        for i, view in enumerate(view_list):
            # print('Evaluating:', scan, 'View:', view)
            predicted_rgb_path = os.path.join(rgb_dir, f"{i:08d}.jpg")
            gt_rgb_path = os.path.join(gt_rgb_dir, f"{view:06d}.jpg")

            assert os.path.exists(
                predicted_rgb_path
            ), f"File not found: {predicted_rgb_path}"
            assert os.path.exists(gt_rgb_path), f"File not found: {gt_rgb_path}"

            predicted_rgb = cv2.imread(predicted_rgb_path)
            predicted_rgb = cv2.cvtColor(predicted_rgb, cv2.COLOR_BGR2RGB)
            gt_rgb = cv2.imread(gt_rgb_path)
            gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)
            gt_rgb = cv2.resize(
                gt_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
            )  # (600, 800)

            ssim = calculate_ssim(predicted_rgb, gt_rgb)
            psnr = calculate_psnr(predicted_rgb, gt_rgb)

            lpips_value = calculate_lpips(predicted_rgb, gt_rgb)

            scan_ssim.append(ssim)
            scan_lpips.append(lpips_value)
            scan_psnr.append(psnr)
        all_ssim.append(np.mean(scan_ssim))
        all_lpips.append(np.mean(scan_lpips))
        all_psnr.append(np.mean(scan_psnr))

    df = pd.DataFrame(
        zip(scans, all_psnr, all_ssim, all_lpips),
        columns=["Scan", "PSNR", "SSIM", "LPIPS"],
    )
    print(df)
    print(df[df.columns[1:]].mean())
