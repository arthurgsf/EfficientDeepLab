import numpy as _np

def mean_std_norm(x):
    """
    Normalizes x by subtracting by the mean and dividing by std.
    """
    x = (x - _np.mean(x))/_np.std(x)
    return x

def center_crop(image, mask, shape):
    """
    Crops the image from the center. the final image shape is equal to the "shape" parameter
    """
    center = image.shape
    w = shape[0]
    h = shape[1]
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    crop_img = image[int(y):int(y+h), int(x):int(x+w)]
    crop_label = mask[int(y):int(y+h), int(x):int(x+w)]
    return crop_img, crop_label