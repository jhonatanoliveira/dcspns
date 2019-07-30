"""Utilities for SPN learning and inference."""
import numpy as np
import imageio
from skimage.transform import resize
from skimage.util import crop
import time
from collections import deque
from os import listdir
from os.path import isfile, join
import sklearn.mixture

from dcspn import MARG_VAR_VAL


class Database:
    """Common tasks related to an image database."""

    def __init__(self, data=None, dataset_name=None, data_path=None,
                 normalize=False, seed=1234):
        """
        Load given database or get it from sklearn repository.

        Parameters
        ----------
        data: numpy array
            Database as an array.
        dataset_name: str
            Name of database to load.
        data_path: str
            Path to folder containing the image dataset
        """
        self.seed = seed
        self.data = None
        if data is not None:
            self.data = data
        elif dataset_name is not None:
            if dataset_name == "olivetti":
                from sklearn.datasets import fetch_olivetti_faces
                database = fetch_olivetti_faces(
                    data_home="databases",
                    shuffle=True, random_state=self.seed)
                self.data = database["images"]
            else:
                ValueError("Sklearn database not supported.")
        elif data_path is not None:
            data_list = []
            pictures = listdir(data_path)
            pictures.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for file in pictures:
                file_path = join(data_path, file)
                if file_path and file_path.endswith((".png", ".jpeg", ".jpg")):
                    file_arr = imageio.imread(file_path)
                    data_list.append(file_arr)
            data_stack = np.stack(data_list, axis=0)
            data_stack = np.float32(data_stack)
            data_stack = data_stack - data_stack.min()
            data_stack /= data_stack.max()
            random_state = np.random.RandomState(self.seed)
            order = random_state.permutation(len(data_stack))
            self.data = data_stack[order]
        else:
            ValueError("Please, either provide a database or choose one.")
        # Save info about database
        self.batch_size = self.data.shape[0]
        self.img_height = self.data.shape[1]
        self.img_width = self.data.shape[2]
        self.img_channel = self.data.shape[3] if len(
            self.data.shape) > 3 else 1
        # By default, normalize database
        self.has_normalized = False
        if normalize:
            self.normalize()

    def resize(self, new_height, new_width):
        """
        Resize image database.

        Parameters
        ----------
        new_height: int
        new_width: int
        """
        if new_height != self.img_height or new_width != self.img_width:
            resized_database = np.zeros(
                (self.batch_size, new_height, new_width), dtype=np.float32)
            for i in range(self.batch_size):
                resized_database[i, :, :] = resize(
                    self.data[i, :, :],
                    (new_width, new_height), mode="reflect")
            self.data = resized_database
            self.img_height = new_height
            self.img_width = new_width

    def crop(self, crop_width):
        """
        Crop image database.

        Parameters
        ----------
        crop_width: list of lists
            Number of values to remove from the edges of each axis.
            ((before_1, after_1),... (before_N, after_N))

        Example
        -------
        crop( [ [0, 0], [6, 6] ] )
        """
        new_height = self.img_height - (crop_width[0][0] + crop_width[0][1])
        new_width = self.img_width - (crop_width[1][0] + crop_width[1][1])
        resized_database = np.zeros((self.batch_size, new_height, new_width))
        for i in range(self.batch_size):
            resized_database[i, :, :] = crop(
                self.data[i, :, :], crop_width)
        self.data = resized_database
        self.img_height = new_height
        self.img_width = new_width

    def normalize(self):
        """Normalize data to yield zero mean and unit variance."""
        self.data_mean = np.mean(self.data, axis=0, dtype=np.float32)
        self.data_std = np.std(self.data, axis=0, dtype=np.float32)
        # Treat case when std can be zero
        self.data_std[self.data_std == 0] = 1
        self.data = (self.data - self.data_mean) / self.data_std

        self.has_normalized = True

    def de_normalize(self, data, out_nhwc=False):
        """
        Transform data from original database using global mean and variance.

        Parameters
        ----------
        data: numpy array
            Data from or relates to the original database
        """
        if not self.has_normalized:
            raise ValueError("Database was not normalized.")
        _data = None
        if not out_nhwc:
            _data = np.reshape(data, (data.shape[0], data.shape[1],
                               data.shape[2]))
        else:
            _data = data
        return _data * self.data_std + self.data_mean

    def split(self, train_percentage=0.9, nhwc_format=True):
        """
        Split into train and validation databases.

        Parameters
        ----------
        train_percentage: float
            Percentage of the database reserved for training
        nhwc_format: boolean
            If true, reshape the database to [batch size, height,
            width, channel] format.
        """
        train_batch_size = int(self.batch_size * train_percentage)
        eval_batch_size = self.batch_size - train_batch_size
        train_data = self.data[0:train_batch_size, :, :]
        eval_data = self.data[
            train_batch_size:train_batch_size + eval_batch_size, :, :]
        # Reshape for (N,H,W,C) format
        train_data = np.reshape(
            train_data,
            (train_batch_size, self.img_height,
             self.img_width, self.img_channel))
        eval_data = np.reshape(
            eval_data,
            (eval_batch_size, self.img_height,
             self.img_width, self.img_channel))
        return train_data, eval_data

    @staticmethod
    def marginalize(data, ver_idxs, hor_idxs):
        """
        Set values in data for marginalization constant.

        Data is internally copied.

        Parameters
        ----------
        data: numpy array
            Data to set marginalization constant
        ver_idx: list of list
            Vertical indexes where pixels should become the marginalization
             constant. For example: [[0, 50], [80, 100]]
        hor_idx: list of list
            Horizontal indexes where pixels should become the marginalization
             constant. For example: [[0, 50], [80, 100]]
        """
        if len(hor_idxs) != len(ver_idxs):
            raise ValueError("Both indexes lists should have the same length.")

        marg_data = data.copy()
        for patch in range(len(hor_idxs)):
            ver_start = ver_idxs[patch][0]
            ver_stop = ver_idxs[patch][1]
            hor_start = hor_idxs[patch][0]
            hor_stop = hor_idxs[patch][1]
            marg_data[:, ver_start:ver_stop,
                      hor_start:hor_stop, :] = MARG_VAR_VAL
        return marg_data

    @staticmethod
    def means_stds_quantiles(data, num_components):
        """
        Mean and standard deviation from a 2D-1 channel database of images.

        Parameters
        ----------
        data: numpy array
            Data to be computed
        num_components : int
            Quantity of quantiles to divide the data
        """
        if data.shape[3] != 1:
            raise ValueError("Only works for image\
                with 1 channel (grey-scale).")

        params_shape = (data.shape[1], data.shape[2], num_components)
        leaf_means = np.zeros(params_shape)
        leaf_stds = np.zeros(params_shape)
        sorted_data = np.sort(data, axis=0)
        quantile_size = data.shape[0] / num_components
        for k in range(num_components):
            lower_idx = int(k * quantile_size)
            upper_idx = int((k + 1) * quantile_size)
            slice_data = sorted_data[lower_idx:upper_idx, :, :, :]
            leaf_means[:, :, k] = np.reshape(
                np.mean(slice_data, axis=0),
                (params_shape[0], params_shape[1]))
            _std = np.std(slice_data, axis=0)
            _std[_std == 0] = 1
            leaf_stds[:, :, k] = np.reshape(
                _std, (params_shape[0], params_shape[1]))
        return leaf_means, leaf_stds

    @staticmethod
    def means_stds_gmms(data, num_components):
        """
        Mean and standard deviation from a RGB database of images.
        The mean and std are derived from the gaussian components in
        a gaussian mixture model learned for each pixel.

        Parameters
        ----------
        data: numpy array
            Data to be computed
        num_components : int
            Quantity of components for the mixture model
        """
        if len(data.shape) != 4:
            raise ValueError("Only works for batch\
                of images. Data must be have 4 shapes.")

        # Initializations
        params_shape = (data.shape[1], data.shape[2], num_components,
                        data.shape[3])
        leaf_means = np.zeros(params_shape)
        leaf_stds = np.zeros(params_shape)
        # For each pixel, compute GMM and extract
        # means and covariances from gaussians
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                gmm = sklearn.mixture.GaussianMixture(
                    n_components=num_components, covariance_type='diag')
                gmm.fit(data[:, i, j, :])
                leaf_means[i, j, :, :] = gmm.means_
                leaf_stds[i, j, :, :] = gmm.covariances_

        return leaf_means, leaf_stds

    @staticmethod
    def mean_square_error(data, data_target, save_imgs_path=None):
        """
        Compute MSE between two image databases.

        data: numpy array
            Computed data with shape [N, H, W, C]
        data_target: numpy array
            Evaluation data with shape [N, H, W, C]
        save_imgs_path: str
            Path to folder where images and MSE log should be saved.
            Path should not include the "/" at the end.
        """
        # MSE needs data as 2D images
        if len(data.shape) == 4:
            data = np.squeeze(data)
            data_target = np.squeeze(data_target)

        # Simple MSE function
        def _mse(d1, d2):
            # Shape of d1 and d2 should be [H, W]
            diff_arr = ((255 * d2).astype(dtype="int") -
                        (255 * d1).astype(dtype="int")) ** 2
            mse = np.mean(diff_arr[diff_arr != 0])
            return mse
        if save_imgs_path is None:
            total_mse = 0
            amt = 0
            for img in range(data.shape[0]):
                total_mse += _mse(data[img, :, :], data_target[img, :, :])
                amt += 1
            return total_mse / amt
        else:
            mse_img = {
                _mse(data[img_idx, :, :],
                     data_target[img_idx, :, :]): img_idx
                for img_idx in range(data.shape[0])
            }
            log_mse = open("{}/mse_log.txt".format(save_imgs_path), "w")
            total_mse = 0
            amt = 0
            for idx, img_mse in enumerate(reversed(sorted(mse_img.keys()))):
                Database.save_image(
                    "{}/{}.png".format(save_imgs_path, idx),
                    data[mse_img[img_mse], :, :])
                total_mse += img_mse
                log_mse.write("{},{}\n".format(idx, img_mse))
                amt += 1
            final_mse = total_mse / amt
            log_mse.write("Total, {}".format(final_mse))
            log_mse.close()
            return final_mse

    @staticmethod
    def save_image(im_path, im_data):
        """Save numpy array to image."""
        try:
            imageio.imsave(im_path, (255 * im_data).astype(dtype="uint8"))
            np_path = im_path[:im_path.find(".")] + "_mpe.npy"
            np.save(np_path, im_data, allow_pickle=False)
        except Exception as e:
            print("Could not show or save image(s)")
            print(e)


def save_images(name, images, path):
    """Save multiple images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=3)

    amt = int(np.ceil(np.sqrt(images.shape[0])))
    r, c = amt, amt

    fig, axs = plt.subplots(r, c, subplot_kw=dict(polar=True))
    axs = np.array([[axs]]) if not type(axs) is list else axs
    cnt = 0
    for i in range(r):
        for j in range(c):
            if cnt < images.shape[0]:
                axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
    fig.savefig("{}/{}.png".format(path, name))
    plt.close()


def plot_cost_graph(name, costs, dest_img):
    """Plot cost graph and save image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(range(1, len(costs) + 1), costs)
    plt.title('{}'.format(name))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.savefig(dest_img)
    plt.close()


def random_mini_batches(x, y=None, mini_batch_size=64, seed=1234):
    """
    Create a list of random minibatches from (X, Y).

    Parameters
    ----------
    X : numpy array
        input data, of shape (input size, number of examples) (m, Hi, Wi)
    mini_batch_size : int
        size of the mini-batches, integer
    seed : int
        this is only for the purpose of grading,
        so that you're "random minibatches are the same as ours.

    Return
    -------
    mini_batches : list
        list of synchronous (mini_batch_x)
    """
    # number of training examples
    m = x.shape[0]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    random_state = np.random.RandomState(seed)
    permutation = list(random_state.permutation(m))
    shuffled_x = x[permutation, :, :]
    shuffled_y = None
    if y is not None:
        shuffled_y = y[permutation, :, :]

    # Step 2: Partition (shuffled_x, shuffled_y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[
            k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch = None
        if y is not None:
            mini_batch_y = shuffled_y[
                k * mini_batch_size: k * mini_batch_size + mini_batch_size,
                :, :]
            mini_batch = (mini_batch_x, mini_batch_y)
        else:
            mini_batch = (mini_batch_x, None)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[
            num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch = None
        if y is not None:
            mini_batch_y = shuffled_y[
                num_complete_minibatches * mini_batch_size: m, :, :]
            mini_batch = (mini_batch_x, mini_batch_y)
        else:
            mini_batch = (mini_batch_x, None)
        mini_batches.append(mini_batch)

    return mini_batches


class TicTac:
    """Simple class for timing execution."""

    def __init__(self):
        """Constructor."""
        self.timers = {}
        self.timers_ids_stack = []

    def tic(self):
        """Start timer."""
        if len(self.timers_ids_stack) != 0:
            new_id = 1 + self.timers_ids_stack[len(self.timers_ids_stack) - 1]
            self.timers_ids_stack.append(new_id)
            self.timers[new_id] = [time.perf_counter()]
        else:
            self.timers_ids_stack.append(0)
            self.timers[0] = [time.perf_counter()]

    def tac(self):
        """Stop the closest previously started timer."""
        if len(self.timers_ids_stack) != 0:
            current_id = self.timers_ids_stack.pop()
            self.timers[current_id].append(time.perf_counter())
            return (self.timers[current_id][1] - self.timers[current_id][0])


def kahn_topsort(graph):
    """
    Kahn algorithm for topological sort.

    The idea of Kahn algorithm is to repeatedly remove nodes
    that have zero in-degree.
    This code is inspired from [1].

    Reference
    ---------
    [1] https://algocoding.wordpress.com/2015/04/05/topological-sorting-python/
    """
    # determine in-degree of each node
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            if v in in_degree:
                in_degree[v] += 1
    # collect nodes with zero in-degree
    q = deque()
    for u in in_degree:
        if in_degree[u] == 0:
            q.appendleft(u)
    # list for order of nodes
    l = []
    while q:
        # choose node of zero in-degree and 'remove' it from graph
        u = q.pop()
        l.append(u)
        for v in graph[u]:
            if v in in_degree:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.appendleft(v)
    # if there is a cycle, then return an empty list
    if len(l) == len(graph):
        return l
    else:
        return []


class RAWreader(object):
    """
    Read raw files from dataset.

    Example
    -------
    >>> array_converted = RAWreader(
    >>>     "databases/caltech/Faces_easy",".raw.rescale")
    >>> imgs = array_converted.get_array_of_images()
    """

    def __init__(self, path, extension):
        """Constructor."""
        self.path = path
        self.extension = extension
        self.list_files = [f for f in listdir(path) if isfile(join(path, f))]

    def check_width(self, image):
        """Check width."""
        ini_len = len(image[0])
        for row in image:
            if len(row) != ini_len:
                print(len(row))
                print(ini_len)
                return False

        return True

    def check_path_str(self, path):
        """Check path."""
        if path[-1] != "/":
            path = path + "/"
        return path

    def get_array_of_images(self):
        """Array of images."""
        arr_images = []

        for item_name in self.list_files:

            if item_name.endswith(self.extension):
                pass
            else:
                break

            with open(self.check_path_str(
                    self.path) + item_name, 'r', errors='ignore') as f:
                file = f.read()
                file = file.split("\n")

                arr = []
                for line in file:
                    if len(line) > 0:
                        x = line.strip(" ").strip("\n").split(" ")
                        x = [int(i) for i in x]
                        arr.append(x)
                if arr[-1] == ['']:
                    # remove last item which is empty list [[]]
                    arr = arr[:-1]

                if self.check_width(arr):
                    m_arr = np.array(arr)
                    arr_images.append(m_arr)
                else:
                    raise ValueError('Pixels lost in conversion')

        output = np.stack(arr_images)

        return output
