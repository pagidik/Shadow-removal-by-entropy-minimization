'''
Name: Sumegha Singhania, Kishore Reddy Pagidi
Date: 10.13.22
Class: CS 7180 - Advanced Perception

Paper implemented: G.D. Finlayson, M.S. Drew and C. Lu, 
"Intrinsic Images by Entropy Minimisation", Proc. European Conf. Computer Vision, 2004

This file contains the executable that calls functions defined in Project2_functions.py
'''
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from Project2_functions import chrom_case_a,chrom_case_b,chrom_case_c, get_log_chrom, projections_case_a, invariant_image_case_a,projections,invariant_image, L1,plot_entropies

if __name__ == "__main__":
    path = input("Enter path to image?")
    img = cv2.imread(path)
    img64 = cv2.GaussianBlur(img.astype(np.float64), (3, 3), 1)
    cv2.imshow("Original Image",img)

    case = input("Enter chromaticity type:\n"
        "A => Divide by single color channel\n"
        "B => Divide by arithmetic mean\n"
        "C => Divide by geometric mean\n")

    if (case=='A'):
        c = input ("Enter color channel: B/G/R\n")
        chromaticity = chrom_case_a(img,c)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        # invariant_angle = projections_case_a(img,log_chromaticity,c)
        invariant_angle,entropy_values = projections(img,log_points)
        # invariant_img = invariant_image_case_a(log_chromaticity,invariant_angle,c)
        invariant_img = invariant_image(log_points,invariant_angle)
        min_entropy_angle = invariant_angle - np.pi/2
        l1_img = L1(img,log_points,min_entropy_angle)

    elif(case=='B'):
        chromaticity = chrom_case_b(img)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        invariant_angle,entropy_values = projections(img,log_points)
        invariant_img = invariant_image(log_points,invariant_angle)
        min_entropy_angle = invariant_angle - np.pi/2
        l1_img = L1(img,log_points,min_entropy_angle)

    else:
        chromaticity = chrom_case_c(img64)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        invariant_angle,entropy_values = projections(img64,log_points)
        invariant_img = invariant_image(log_points,invariant_angle)
        min_entropy_angle = invariant_angle - np.pi/2
        l1_img = L1(img64,log_points,min_entropy_angle)
        plot_entropies(entropy_values)
        
    cv2.waitKey()

