'''
Name: Sumegha Singhania, Kishore Reddy Pagidi
Date: 10.13.22
Class: CS 7180 - Advanced Perception

Paper implemented: G.D. Finlayson, M.S. Drew and C. Lu, 
"Intrinsic Images by Entropy Minimisation", Proc. European Conf. Computer Vision, 2004

This file contains all functions corresponding to the various processes required for the process. 

We broke down the process into the following steps and made the functions accordingly. 
There were a few parts we were stuck at, especially with L1 difference, we used 
an online source for the same for better understanding. The source has been mentioned
above the function. 

Steps:
1. Get image
2. Find chromaticity
3. Get log chromaticity
4. Project on axis form 0 to 180 degrees
5. Make a histogram
6. Find minimum entropy
7. Find invariant image
8. Find L1 difference

We attempted to find invariant image for 3 cases:
Case A: Divide by single color channel and find log chromaticity
Case B: Divide by arithmetic mean
Case C: Divide by geometric mean

Some functions below: projections_case_a(), invariant_image_case_a() were derived from 
our understanding of the process, but they do not give expected results as obtained by 
the projections(). They have been left in here, but are commented out in the main function. 
This is described in further detail in the report. 
'''

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

U = np.array([ [1/math.sqrt(2), -1/math.sqrt(2), 0], [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)] ]) # orthogonal matrix

def projectOntoPlane(log_points, orthMatrix):
    return log_points @ orthMatrix.T

# Step 5: Make histogram
# Step 6: Find minimum entropy
def ShannonEntropy(I, bandwidth = 1):
    nbins = round((np.max(I) - np.min(I)) / bandwidth)
    P = np.histogram(I, nbins)[0] / I.size
    P = P[P != 0]
    return -np.sum(P * np.log2(P))

# Step 2: Find 2D chromaticity values
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def chrom_case_a(img,c):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0

    if (c=='R'):
        divisor = red
    elif(c=='B'):
        divisor = blue
    else:
        divisor = green

    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom

    # Case B: Dividing by mean of colour channels
def chrom_case_b(img):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0

    divisor = red+blue+green+1
    divisor[divisor==0] = 1.0

    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom

    # Case C: Dividing by  geometric mean of colour channels
def chrom_case_c(img):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0

    divisor = np.power((red * green * blue), 1.0/3)
    divisor[divisor==0] = 1.0

    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom


# Step 3: Get log chromaticity values (same for all 3 cases)
def get_log_chrom(chrom):
    log_chrom = np.zeros_like(chrom, dtype = np.float64)
    log_chrom[:, :, 0] = np.log(chrom[:,:,0])
    log_chrom[:, :, 1] = np.log(chrom[:,:,1])
    log_chrom[:, :, 2] = np.log(chrom[:,:,2])

    # display the log chromaticity image
    cv2.imshow("Log_chrom",log_chrom)

    # convert into 3d points
    log_chrom_b = np.atleast_3d(log_chrom[:, :, 0])
    log_chrom_g = np.atleast_3d(log_chrom[:, :, 1])
    log_chrom_r = np.atleast_3d(log_chrom[:, :, 2])

    # join the log values to get 3d points representing log_r,log_g,log_b values
    log_points = np.concatenate((log_chrom_r,log_chrom_g,log_chrom_b),axis=2)

    return log_chrom, log_points

# Step 4: Find point projections for all angles
# Step 5,6: Find min entropy and corresponding invariant angle
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def projections_case_a(img,log_chrom,c):
    
    img_shape = img.shape[0]*img.shape[1]

    # initialising a list of entropies for angles from 0-180
    entropy_values = np.zeros(181, dtype = np.float64)

    # converting the angles to radians
    radians = np.radians(np.linspace(0,180,181))

    for i,theta in enumerate(radians):
        # I represents the projection of the 2D chromaticity point on a line with angle theta
        if (c=='R'):
            I = log_chrom[:,:,0]*np.cos(theta)+log_chrom[:,:,1]*np.sin(theta)     # (log b/r and log g/r)
        elif (c=='B'):
            I = log_chrom[:,:,1]*np.cos(theta)+log_chrom[:,:,2]*np.sin(theta)     # (log g/b and log r/b)
        else:
            I = log_chrom[:,:,0]*np.cos(theta)+log_chrom[:,:,2]*np.sin(theta)     # (log b/g and log r/g)

        # The following steps help clip the pixel values
        # between 5-95% to adjust for noise
        Mean = np.mean(I)
        Std_dev = np.std(I)
        lower_bound = Mean-(3*Std_dev)
        upper_bound = Mean+(3*Std_dev)
        clipped_values = np.clip(I,lower_bound,upper_bound)

        # calculating histogram bin size using the formula presented in paper
        bin_size = 3.5*Std_dev*img_shape**(-1/3)

        # calculating entropy value for the angle
        entropy_values[i] = ShannonEntropy(clipped_values,bin_size)

    # Find minimum entropy value and corresponding angle which will be the invariant angle
    min_entropy = np.min(entropy_values)
    min_ent_index = np.argmin(entropy_values)
    min_entropy_angle = radians[min_ent_index]
    invariant_angle = min_entropy_angle+ np.pi/2
    print ("Minimum Entropy: ",min_entropy)
    print ("Invariant Angle: ", invariant_angle)
    return invariant_angle,entropy_values

    # Case B,C
def projections(img,log_points):
    img_shape = img.shape[0]*img.shape[1]
    projected_points = projectOntoPlane(log_points, U)

    # initialising a list of entropies for angles from 0-180
    entropy_values = np.zeros(181, dtype = np.float64)

    # converting the angles to radians
    radians = np.radians(np.linspace(0,180,181))

    for i,theta in enumerate(radians):
        I = projected_points[:,:,0]*np.cos(theta) + projected_points[:,:,1]*np.sin(theta)
        # The following steps help clip the pixel values
        # between 5-95% to adjust for noise
        Mean = np.mean(I)
        Std_dev = np.std(I)
        lower_bound = Mean+3*(-Std_dev)
        upper_bound = Mean+3*(+Std_dev)
        clipped_values = np.clip(I,lower_bound,upper_bound)

        # calculating histogram bin size using the formula presented in paper
        bin_size = 3.5*Std_dev*img_shape**(-1/3)

        # calculating entropy value for the angle
        entropy_values[i] = ShannonEntropy(clipped_values,bin_size)

    # Find minimum entropy value and corresponding angle which will be the invariant angle
    min_entropy = np.min(entropy_values)
    min_ent_index = np.argmin(entropy_values)
    min_entropy_angle = radians[min_ent_index]

    # To find angle of log chromaticity values
    invariant_angle = min_entropy_angle+ np.pi/2
    print ("Minimum Entropy: ",min_entropy)
    print ("Invariant Angle (degrees): ", np.rad2deg(invariant_angle))
    print ("Minimum entropy angle(degrees): ",np.rad2deg(min_entropy_angle))
    return invariant_angle,entropy_values

# Step 7: Find invariant image
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def invariant_image_case_a(log_chrom,invariant_angle,c):
    if (c=='R'):
        I = log_chrom[:,:,0]*np.cos(invariant_angle)+log_chrom[:,:,1]*np.sin(invariant_angle)     # (log b/r and log g/r)
    elif (c=='B'):
        I = log_chrom[:,:,1]*np.cos(invariant_angle)+log_chrom[:,:,2]*np.sin(invariant_angle)     # (log g/b and log r/b)
    else:
        I = log_chrom[:,:,0]*np.cos(invariant_angle)+log_chrom[:,:,2]*np.sin(invariant_angle)     # (log b/g and log r/g)

    # Finding inverse of log values
    I = np.exp(I)
    cv2.imshow("Invariant image",I)
    return I

    # Case B,C
def invariant_image(log_points,invariant_angle):
    projected_points = projectOntoPlane(log_points,U)
    I = projected_points[:,:,0]*np.cos(invariant_angle) + projected_points[:,:,1]*np.sin(invariant_angle)
    I = np.exp(I)
    cv2.imshow("Invariant image",I)
    return I

# defining L1 difference. 
# Source: https://github.com/matkovst/ComputerVisionToolkit/blob/a1dc765e68ecb3174d919c959a4c8c84bfdd950a/samples/Shadow-removal/_experimental.py 
def L1(img, Rho, minEntropyAngle):
    N = img.shape[0] * img.shape[1]
    Chi = projectOntoPlane(Rho, U)
    e = np.array([-1 * math.sin(minEntropyAngle), math.cos(minEntropyAngle)])
    eT = np.array([np.cos(minEntropyAngle), np.sin(minEntropyAngle)])
    Ptheta = np.outer(eT, eT)
    Chitheta = Chi @ Ptheta.T
    I = Chi @ e
    Itheta = Chitheta @ e

    IMostBrightest = np.sort( I.reshape(I.shape[0] * I.shape[1]) )
    IMostBrightest = IMostBrightest[IMostBrightest.size - int(0.01*math.ceil(N)) : IMostBrightest.size]
    IMostBrightestTheta = np.sort( Itheta.reshape(Itheta.shape[0] * Itheta.shape[1]) )
    IMostBrightestTheta = IMostBrightestTheta[IMostBrightestTheta.size - int(0.01*math.ceil(N)) : IMostBrightestTheta.size]
    ChiExtralight = (np.median(IMostBrightest) - np.median(IMostBrightestTheta)) * e
    Chitheta += ChiExtralight

    Rhoti = Chitheta @ U
    cti = np.exp(Rhoti)
    ctiSum = np.sum(cti, axis = 2)
    ctiSum = ctiSum.reshape(cti.shape[0], cti.shape[1], 1)
    rti = cti / ctiSum

    cv2.imshow("I1D", cv2.normalize(rti, 0, 255, cv2.NORM_MINMAX))

# function to plot entropies
def plot_entropies(entropy_values):
    fig = plt.figure()
    ent_plt = fig.add_subplot(111)
    ent_plt.set_title('Entropy values')
    ent_plt.set_xlabel('Angle (in degrees)')
    ent_plt.set_ylabel('Entropy')
    ent_plt.plot(entropy_values,label = 'observed entropy')
    ent_plt.legend()
    plt.show()



            
