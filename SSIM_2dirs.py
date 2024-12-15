import os
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def compute_ssim_for_dirs(dir1, dir2, output_file="ssim_results.txt"):
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))

    if len(files1) != len(files2):
        print("Warning: The two directories do not have the same number of files!")

    with open(output_file, "w") as f:
        total_ssim = 0
        count = 0

        for file1, file2 in zip(files1, files2):
            path1 = os.path.join(dir1, file1)
            path2 = os.path.join(dir2, file2)

            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"Skipping {file1} and {file2}: Could not load one of the images.")
                continue

            if img1.shape != img2.shape:
                print(f"Skipping {file1} and {file2}: Images have different dimensions.")
                continue

            ssim_score, _ = ssim(img1, img2, full=True)
            f.write(f"{file1} - {file2}: {ssim_score:.4f}\n")

            print('ssmin score: ', ssim_score)
            total_ssim += ssim_score
            count += 1

        if count > 0:
            avg_ssim = total_ssim / count
            f.write(f"\nAverage SSIM: {avg_ssim:.4f}\n")
            print(f"Average SSIM: {avg_ssim:.4f}")
        else:
            print("No valid image pairs to compute SSIM.")

dir1 = "/Users/sigvard/Desktop/FFHQ"
dir2 = "/Users/sigvard/Desktop/Results FFHQ/Colorization FFHQ /Scale 0.1"
compute_ssim_for_dirs(dir1, dir2, output_file="ssim_results.txt")
