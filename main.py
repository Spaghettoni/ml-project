from PIL import Image, ImageEnhance
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from matplotlib import colors
from mpl_toolkits import mplot3d
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def showSaturation(image):
    plt.subplot(111)
    plt.imshow(image)
    plt.show()
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


def masking(image, show_plots=False):
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if show_plots:
        showSaturation(img_hsv)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([8, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 255
    output_img[np.where(mask != 0)] = 0

    if show_plots:
        plt.subplot(221)
        plt.imshow(mask0)
        plt.subplot(222)
        plt.imshow(mask1)
        plt.subplot(223)
        plt.imshow(mask)
        plt.subplot(224)
        plt.imshow(output_img)
        plt.show()

    return mask, output_img


def kmeans(image, k, attempts):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(twoDimage, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


def thresholding(image, show_plots=False):
    img = cv2.imread(image)
    B, G, R = cv2.split(img)

    if show_plots:
        plt.subplot(221)
        plt.imshow(R)
        plt.subplot(222)
        plt.imshow(B)
        plt.subplot(223)
        plt.imshow(G)
        plt.subplot(224)
        plt.imshow(R*3-G*3)
        plt.show()

    merged = cv2.merge([B-B, G-G, (R-G)*3])
    rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
    _, thresh = cv2.threshold(rgb, np.mean(rgb), 255, cv2.THRESH_BINARY)
    result = cv2.Canny(thresh, 200, 300)

    if show_plots:
        plt.subplot(221)
        plt.imshow(rgb)
        plt.subplot(222)
        plt.imshow(thresh)
        plt.subplot(224)
        plt.imshow(result)
        plt.show()

    _watershed(result, True)
    return result


def _watershed(image, show_plots=False):
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    if show_plots:
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Overlapping objects')
        ax[1].imshow(-distance, cmap=plt.cm.gray)
        ax[1].set_title('Distances')
        ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
        ax[2].set_title('Separated objects')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    datasets = ['dataset0', 'dataset1']
    for i, dataset in enumerate(datasets):
        # if i == 0:
        #     continue
        for name in os.listdir(dataset):
            if i == 1 and name[-9::1] in ['masks.tif', 'annot.csv']:
                continue
            file = dataset + '/' + name
            file_norm = dataset + '_norm/' + name
            kmeans_file_out = 'out' + str(i) + '/kmeans/' + name
            watershed_file_out = 'out' + str(i) + '/watershed/' + name
            masking_file_out = 'out' + str(i) + '/masking/' + name

            print(file)

            # resize images to height of 300 pixels
            im = Image.open(file)
            im.thumbnail((300, 300), Image.ANTIALIAS)
            im.save(file_norm, "JPEG")

            # 1. method - k_means
            # seg = kmeans(file_norm, 2, 10)
            # im = Image.fromarray(seg)
            # im.save(kmeans_file_out)

            # 2. method - watershed
            # res = thresholding(file_norm, True)
            # im = Image.fromarray(res)
            # im.save(watershed_file_out)

            # 3.method - masking
            _, res = masking(file_norm, True)
            im = Image.fromarray(res)
            im.save(masking_file_out)


            break
