import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps


def view_image(img, filename = 'image'):
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(img.reshape(96, 96).squeeze())
    ax.axis('off')

    plt.savefig(filename + '.png')


def convert_to_PIL(img):
    img_r = img.reshape(96, 96)

    pil_img = Image.new('RGB', (96, 96), 'white')
    pixels = pil_img.load()

    for i in range(0, 96):
        for j in range(0, 96):
            if img_r[i, j] > 0:
                pixels[j, i] = (255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255))

    return pil_img


def convert_to_np(pil_img):
    pil_img = pil_img.convert('RGB')

    img = np.zeros((96, 96))
    pixels = pil_img.load()

    for i in range(0, 96):
        for j in range(0, 96):
            img[i, j] = 1 - pixels[j, i][0] / 255

    return img


def crop_image(image):
    cropped_image = image

    # get image size
    width, height = cropped_image.size

    # get image pixels
    pixels = cropped_image.load()

    image_strokes_rows = []
    image_strokes_cols = []

    # run through the image
    for i in range(0, width):
        for j in range(0, height):
            # save coordinates of the image
            if (pixels[i,j][3] > 0):
                image_strokes_cols.append(i)
                image_strokes_rows.append(j)

    # if image is not empty then crop to contents of the image
    if (len(image_strokes_rows)) > 0:
        # find the box for image
        row_min = np.array(image_strokes_rows).min()
        row_max = np.array(image_strokes_rows).max()
        col_min = np.array(image_strokes_cols).min()
        col_max = np.array(image_strokes_cols).max()

        # find the box for cropping
        margin = min(row_min, height - row_max, col_min, width - col_max)

        # crop image
        border = (col_min, row_min, width - col_max, height - row_max)
        cropped_image = ImageOps.crop(cropped_image, border)

    # get cropped image size
    width_cropped, height_cropped = cropped_image.size

    # create square resulting image to paste cropped image into the center
    dst_im = Image.new("RGBA", (max(width_cropped, height_cropped), max(width_cropped, height_cropped)), "white")
    offset = ((max(width_cropped, height_cropped) - width_cropped) // 2, (max(width_cropped, height_cropped) - height_cropped) // 2)
    # paste to the center of a resulting image
    dst_im.paste(cropped_image, offset, cropped_image)

    #resize
    dst_im.thumbnail((96, 96), Image.ANTIALIAS)

    return dst_im


def normalize(arr):
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


def normalize_image(image):
    arr = np.array(image)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    return new_img


def alpha_composite(front, back):
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def convert_to_rgb(image):
    image_rgb = alpha_composite_with_color(image)
    image_rgb.convert('RGB')

    return image_rgb
