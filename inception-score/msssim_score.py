import os
import argparse
import sys

import tensorflow as tf
import numpy as np

from skimage.transform import resize
from scipy import signal
from scipy.ndimage.filters import convolve
import glob
import deepdish as dd
import json


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
  
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small 
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
  
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
  
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small 
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.
  
    Returns:
      MS-SSIM score between `img1` and `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
                im1, im2, max_val=max_val, filter_size=filter_size,
                filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))

def load_data_from_h5(fullpath):
    import glob
    import deepdish as dd
    import h5py

    # import pdb; pdb.set_trace()
    h5file = os.path.join(fullpath,)
    print ('read h5 from {}'.format(h5file))
    
    labels = h5py.File(h5file)['classIDs']
    label_keys = list(set(labels))
    print ('find {} labels {}'.format(len(label_keys), label_keys))

  
    data = h5py.File(h5file)['output_256']
    
    images = {l:[] for l in label_keys}
    tot = 0
    print ('find {} images '.format(data.shape[0]))
    for i in range(data.shape[0]):
        img = data[i]
        # import pdb; pdb.set_trace()
        assert((img.shape[0] == 256 or img.shape[0] == 128 or img.shape[0] == 64) and img.shape[2] == 3)
        if not (img.min() >= 0 and img.max() <= 255 and img.mean() > 1):
            print ('WARNING {}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean()))
            continue	
        #assert img.min() >= 0 and img.max() <= 255 and img.mean() > 1, '{}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean())
        images[labels[i]].append(img)

        tot += data.shape[0]
    for k, v in images.items():
        print ('label {} has {} images'.format(k, len(v)))

    return images

def load_data(fullpath):
    print(fullpath)
    images = []
    h5file = glob.glob(os.path.join(fullpath, '*.h5'))
    print ('find {} h5 files'.format(len(h5file)))
    images = {}
    tot = 0
    for hf in h5file:
        name = os.path.splitext(os.path.split(hf)[1])[0]
        label = int(name.split('_')[-1])
        data = dd.io.load(hf)['samples']
        images[label] = [d for d in data]
        tot += data.shape[0]
        # sys.stdout.write('read {} with {} images, '.format(label, data['samples'].shape[0]))
        # sys.stdout.flush()
    
    for k, v in images.items():
        print ('label {} has {} images'.format(k, len(v)))
    return images

def evalute(data):
    labels = data.keys()
    
    res = {}
    for lab in labels:
        ldata = data[lab]
        n = len(ldata)
        
        score = []
        for i in range(n-1):
            for j in range(i, n):
                ssim = MultiScaleSSIM(ldata[i][np.newaxis,:,:,:], ldata[j][np.newaxis,:,:,:])
                score.append(ssim)
        sys.stdout.write('eval {} with {} images score {}, '.format(lab, n, score[-1]))
        sys.stdout.flush()
        sys.stdout.write(".")
        sys.stdout.flush()
        res[lab] = float(np.mean(score))
    return res

if __name__ == "__main__":
    ours_path = '/data/data2/Shared_YZ/Results/birds/eval_bs_1testing_num_10/zz_mmgan_plain_gl_disc_birds_256_G_epoch_500.h5'
    stackgan_path = '/data/data2/Shared_YZ/StackGAN_visual_results/test_large_samples2_29330/'

    stackgan_data = load_data(stackgan_path)
    ours_data = load_data_from_h5(ours_path)
    assert(len(stackgan_data) == len(ours_data))

    print ('results of ours .. ')
    our_score = evalute(ours_data)
    print ('results of stackgan .. ')
    stackgan_score = evalute(stackgan_data)

    json.dump(our_score, open('ours_ssim_score.json','w'), indent=True)
    json.dump(stackgan_score, open('stackgan_ssim_score.json','w'), indent=True)