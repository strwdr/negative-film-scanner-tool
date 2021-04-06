import glob
import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt


def plt_imshow(img):
    img_tmp = img.copy()
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_tmp = (img).astype(np.uint8)
    # set size
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    # convert color from CV2 BGR back to RGB
    image = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def make_mono_from_BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reverse_rgb(image):
    return 255 - image


def equalize_histogram(image):
    equ_image = cv2.equalizeHist(image)
    return equ_image


def equalize_adaptive_histogram(image, clipLimit=2.0, tileGridSize=8):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    equalized = clahe.apply(image)
    return equalized


def sharpen(W, image):
    # print("W = " + str(W) + ":")
    image_laplace = cv2.Laplacian(image, cv2.CV_8U)
    image_out = cv2.addWeighted(image, 1, image_laplace, W, 0)
    return image_out


def border_mask(image):
    image = copy.deepcopy(image)
    mask = image > 70
    image[mask] = 255
    return image


tmp_image = None
out_image = None

title_window = "Modify image"
title_window_preview = "image preview"

rotate_name = 'Rotate'
x_offset = 'x offset'
y_offset = 'y offset'
x_beg = 'x beg'
y_beg = 'y beg'
equ_btn = 'equ tgl'
shr_btn = 'shr tgl'
rev_btn = 'rev tgl'
clip_limit = 'clip lim'
tile_grid_size = 'tile grid'
shr_W = 'shr W'


def on_trackbar(val):
    global out_image

    rotate_deg = cv2.getTrackbarPos(rotate_name, title_window) - 50

    x_offset_val = cv2.getTrackbarPos(x_offset, title_window)
    y_offset_val = cv2.getTrackbarPos(y_offset, title_window)

    x_beg_val = cv2.getTrackbarPos(x_beg, title_window)
    y_beg_val = cv2.getTrackbarPos(y_beg, title_window)

    equ_option = cv2.getTrackbarPos(equ_btn, title_window)
    shr_option = cv2.getTrackbarPos(shr_btn, title_window)
    rev_option = cv2.getTrackbarPos(rev_btn, title_window)

    shr_W_val = -cv2.getTrackbarPos(shr_W, title_window) / 50
    clip_limit_val = cv2.getTrackbarPos(clip_limit, title_window) / 10
    tile_grid_size_val = cv2.getTrackbarPos(tile_grid_size, title_window) + 1

    y = int(x_beg_val * (tmp_image.shape[1] / 200))
    x = int(y_beg_val * (tmp_image.shape[0] / 200))
    w = int(y_offset_val * (tmp_image.shape[0] / 200))
    h = int(x_offset_val * (tmp_image.shape[1] / 200))
    rows, cols = tmp_image.shape

    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rotate_deg, 1)
    output = cv2.warpAffine(tmp_image, M, (cols, rows))
    output = output[x:x + w, y:y + h]

    if rev_option == 1:
        output = reverse_rgb(output)
    if equ_option == 1:
        output = equalize_adaptive_histogram(output, clip_limit_val, tile_grid_size_val)
    if equ_option == 2:
        output = equalize_histogram(output)
    if shr_option == 1:
        output = sharpen(shr_W_val, output)

    out_image = copy.deepcopy(output)
    preview = create_preview(out_image)

    cv2.imshow(title_window_preview, preview)
    # cv2.resizeWindow(title_window, 400, 400)


def create_preview(image, new_width=400, new_height=400):
    tmp = copy.deepcopy(image)
    dsize = ((int)(tmp.shape[1] * (new_height / tmp.shape[0])), new_height)
    output = cv2.resize(tmp, dsize, interpolation=cv2.INTER_AREA)
    dsize = (new_width, (int)(tmp.shape[0] * (new_width / tmp.shape[1])))
    output = cv2.resize(output, dsize, interpolation=cv2.INTER_AREA)
    # output = output[0:0 + 100, 0:0 + 100]
    return output


def run_for_file(image, outdir, index, type='.jpg', naming_base='ph-'):
    global tmp_image
    tmp_image = copy.deepcopy(image)
    tmp_image = make_mono_from_BGR(tmp_image)

    on_trackbar(1)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            path = outdir + "/" + naming_base + str(index) + type
            print('saving to ' + path)
            cv2.imwrite(path, out_image)
            break
        if key == ord('s'):
            print('skipping')
            break


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_for_dir(indir='./input', outdir='./output'):
    ensure_dir(indir)
    ensure_dir(outdir)

    cv2.namedWindow(title_window, cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(title_window_preview, cv2.WINDOW_AUTOSIZE)

    cv2.resizeWindow(title_window, 700, 420)

    cv2.createTrackbar(rotate_name, title_window, 50, 100, on_trackbar)
    cv2.createTrackbar(x_offset, title_window, 200, 200, on_trackbar)
    cv2.createTrackbar(y_offset, title_window, 200, 200, on_trackbar)
    cv2.createTrackbar(x_beg, title_window, 0, 200, on_trackbar)
    cv2.createTrackbar(y_beg, title_window, 0, 200, on_trackbar)

    cv2.createTrackbar(shr_W, title_window, 25, 200, on_trackbar)
    cv2.createTrackbar(clip_limit, title_window, 20, 100, on_trackbar)
    cv2.createTrackbar(tile_grid_size, title_window, 7, 20, on_trackbar)
    cv2.createTrackbar(shr_btn, title_window, 1, 1, on_trackbar)
    cv2.createTrackbar(equ_btn, title_window, 1, 2, on_trackbar)
    cv2.createTrackbar(rev_btn, title_window, 1, 1, on_trackbar)
    index = 1
    for filename in glob.glob(os.path.join(indir, '*.jpg')):
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        run_for_file(image, outdir, index)
        index += 1

    for filename in glob.glob(os.path.join(indir, '*.png')):
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        run_for_file(image, outdir, index)
        index += 1

    for filename in glob.glob(os.path.join(indir, '*.JPG')):
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        run_for_file(image, outdir, index)
        index += 1


if __name__ == '__main__':
    run_for_dir()
