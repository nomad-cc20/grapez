from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray
from skimage import feature as ftr
from skimage import color


class ImagePreProcessor:
    """
    Handles image preprocessing.
    """

    def __init__(self, cell_size: int, block_size: int):
        """
        ImagePreProcessor constructor.
        :param cell_size: square cell size in pixels
        :param block_size: square neighbourhood size in cells
        """
        self.cell_size = cell_size
        self.block_size = block_size

    def serve(self, path: str) -> ndarray:
        """
        Prepares and returns an image of given path as a 2D matrix.
        :param path: the image path
        :return: a list of edges, as long as 9*[cell_count]
        """
        img = plt.imread(path)
        img = color.rgb2gray(img)
        h = ftr.hog(img, orientations=9,
                    pixels_per_cell=(self.cell_size, self.cell_size),
                    cells_per_block=(self.block_size, self.block_size),
                    visualise=False, block_norm='L2-Hys')
        return h
