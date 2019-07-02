"""

Note to user:  This file, while functional, is not fully differentiable in
 PyTorch and is not easily moved to and from the gpu.  For updated version use
 the source copde in modules and updated DiffJPEG module.

"""
# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
import utils


def rgb_to_ycbcr(image):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.],
         [112., -93.786, -18.214]],
        dtype=np.float32).T / 255
    shift = [16., 128., 128.]
    image = image
    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image, torch.from_numpy(matrix), dims=1) + shift
#    result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def rgb_to_ycbcr_jpeg(image):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T
    shift = [0., 128., 128.]
    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image, torch.from_numpy(matrix), dims=1) + shift
#    result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def chroma_subsampling(image):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    image_2 = image.permute(0, 3, 1, 2).clone()
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                            count_include_pad=False)
    cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
    cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
    cb = cb.permute(0, 2, 3, 1)
    cr = cr.permute(0, 2, 3, 1)
    return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


def block_splitting(image):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    k = 8
    height, width = image.shape[1:3]
    batch_size = image.shape[0]
    image_reshaped = image.view(batch_size, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, -1, k, k)


def dct_8x8_ref(image):
    """ Reference Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    image = image - 128
    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * x + 1) * u *
                                          np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        result[u, v] = value
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    return result * scale


def dct_8x8(image):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = scale * torch.tensordot(image, tensor, dims=2)
    #result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def y_quantize(image, rounding, factor=1):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    image = image.float() / (utils.y_table * factor)
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    image = image.float() / (utils.c_table * factor)
    image = rounding(image)
    return image


def compress_jpeg(imgs, rounding=torch.round, factor=1):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    temp = rgb_to_ycbcr_jpeg(imgs*255)
    y, cb, cr = chroma_subsampling(temp)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = block_splitting(components[k])
        comp = dct_8x8(comp)
        comp = c_quantize(comp, torch.round, factor=factor) if k in (
            'cb', 'cr') else y_quantize(comp, torch.round, factor=factor)

        components[k] = comp
    return components['y'], components['cb'], components['cr']
