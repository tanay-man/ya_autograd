# mnist.py
import os
import gzip
import numpy as np
from urllib import request
from tensor import Tensor

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        request.urlretrieve(url, filename)

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return data / 255.0  # normalize

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def load_mnist():
    """
    Downloads and loads MNIST dataset into NumPy arrays.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (X_train, y_train, X_test, y_test)
        X arrays are shape (N, 784), normalized floats.
        y arrays are shape (N,), integers 0-9.
    """
    os.makedirs("data", exist_ok=True)

    for key, url in URLS.items():
        download(url, os.path.join("data", url.split("/")[-1]))

    X_train = load_images("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")
    X_test = load_images("data/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("data/t10k-labels-idx1-ubyte.gz")

    # Return the NumPy arrays directly
    return X_train, y_train, X_test, y_test