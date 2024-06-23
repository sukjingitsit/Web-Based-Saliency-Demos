import numpy as np
from matplotlib import pyplot as plt

def ShowImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
    im = np.transpose(im, (1,2,0))
    plt.axis('off')
    plt.imshow(im)
    plt.title(title)
    plt.show()

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(title)

def VisualizeImageGrayscale(image_3d, percentile=99):
  image_2d = np.sum(np.abs(image_3d), axis=2)
  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)
  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='inferno')
    plt.title(title)

def Visualise(saliency, title = "Saliency Map"):
    if (len(saliency.shape) == 5):
        fig, ax = plt.subplots(saliency.shape[0],saliency.shape[1], figsize=(10*saliency.shape[0],10*saliency.shape[1]))
        for i in range(saliency.shape[0]):
            for j in range(saliency.shape[1]):
                grayscale_mask = VisualizeImageGrayscale(saliency[i][j])
                ShowGrayscaleImage(grayscale_mask, title=title, ax=plt.subplot(saliency.shape[0], saliency.shape[1], i*saliency.shape[1]+j+1))
        return fig
    if (len(saliency.shape) == 4):
        fig, ax = plt.subplots(1,saliency.shape[0], figsize=(10,10*saliency.shape[0]))
        for j in range(saliency.shape[0]):
            grayscale_mask = VisualizeImageGrayscale(saliency[j])
            ShowGrayscaleImage(grayscale_mask, title=title, ax=plt.subplot(1, saliency.shape[0], j+1))
        return fig
    if (len(saliency.shape) == 3):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        grayscale_mask = VisualizeImageGrayscale(saliency)
        ShowGrayscaleImage(grayscale_mask, title=title)
        return fig