import numpy as np
import cv2
import os


def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    height, width = img.shape
    # Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype=np.int)
    img = np.concatenate((pad_left, img), axis=1)

    # Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis=0)

    # Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis=1)

    # Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis=0)
    return img


def tune_image(imt):
    # kernel = np.ones((5, 5), np.uint8)
    max_dimen = imt.shape[0] if imt.shape[0] >= imt.shape[1] else imt.shape[1]
    im_blk = np.zeros((max_dimen, max_dimen), np.uint8)
    d = (im_blk.shape[1] - img.shape[1]) / 2
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            im_blk[x][y + int(d)] = img[x][y]

    #blur = cv2.GaussianBlur(im_blk, (5,5),0)
    small_im = add_padding(im_blk , 4,4,4,4)
    small_im = cv2.resize(small_im, (28, 28))
    cv2.imwrite('result/resized_images/' + filename, small_im)


kernel = np.ones((5, 5), np.uint8)
images = []
images_list = []
size_list = []
folder = 'result/characters/'
for filename in os.listdir(folder):
    size = os.path.getsize(os.path.join(folder, filename))
    size_list.append(size)

size_list = np.array(size_list)

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename), 0)
    # if os.path.getsize(os.path.join(folder, filename)) > 11000:
    #     tune_image(img)
    #     continue
    # else:
    img = cv2.erode(img, kernel, iterations=4)
    img = cv2.blur(img,(3,3))
    th, a = cv2.threshold(img, 80, 255, cv2.THRESH_OTSU)
    if a is not None:
        a = cv2.resize(a, (100, 100))
        # # create blank image - y, x
        col_sum = np.where(np.sum(a, axis=0) > 0)
        row_sum = np.where(np.sum(a, axis=1) > 0)
        y1, y2 = row_sum[0][0], row_sum[0][-1]
        x1, x2 = col_sum[0][0], col_sum[0][-1]
        cropped_image = a[y1:y2, x1:x2]
        #cropped_image = cv2.resize(cropped_image, (64, 64))
        padded_image = add_padding(cropped_image, 4, 4, 4, 4)
        resized_small_img = cv2.resize(padded_image, (28, 28))
        cv2.imwrite('result/resized_images/' + filename, resized_small_img)


print("Images aligned and saved into resized_images folder.")
