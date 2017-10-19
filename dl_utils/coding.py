import numpy as np


def flatBatch2Tensor(batchData, imSize, channels):
    """
        Turns a flattend (square) images into a 4 dimensional tensor

        :param batchData: a tensor shaped [number of images, imsize*imsize*channels]
        :param imSize: the width and height (needs to be squares) of the images
        :param channels: the number of channels
        :return: A numpy array of dimensions batch_size, imSize, imSize, channels

    """
    splitByChannel = [batchData[:, (chan * imSize ** 2):((chan + 1) * imSize ** 2)].reshape((-1, imSize, imSize, 1)) \
                      for chan in range(channels)]
    tensorBatchData = np.concatenate(splitByChannel, 3)
    return tensorBatchData

