import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing

from numpy import genfromtxt, count_nonzero


EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def is_self_supervised_image(filename):
    return filename.endswith("_img.bmp")

def is_self_supervised_label(filename, ext="npy"):
    return filename.endswith("_label_0." + ext)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def split_first_subname(filename, delim='_'):
    return filename.split(delim)[0]

def func(i, filenameGt, filenames, file_root, output_dir, file_format):
    print("Processing ", i, " out of ", len_files)
    if file_format == "npy":
        cur_file = np.load(image_path_city(file_root, filenameGt))
    elif file_format == "csv":
        cur_file = genfromtxt(image_path_city(file_root, filenameGt), delimiter=',', dtype="float32")
    else:
        print("Unsupported file format " + file_format)

    if (cur_file > 0.0).any():
        file_ending = os.path.relpath(filenames[i], file_root)
        fileGt_ending = os.path.relpath(filenameGt, file_root)

        filename_out = os.path.join(output_dir, file_ending)
        file_out_dir = os.path.dirname(filename_out)

        if not os.path.exists(file_out_dir):
            os.makedirs(file_out_dir)

        os.symlink(os.path.join(file_root, filenameGt), filename_out)
        os.symlink(os.path.join(file_root, filenames[i]), os.path.join(output_dir, fileGt_ending))


if __name__ == '__main__':
    file_root = "/mnt/drive_c/datasets/2018_dataset/labelled/"
    output_dir = "/mnt/drive_c/datasets/2018_dataset/filtered_labelled/"

    file_format = "csv"

    filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(file_root), followlinks=False) for f in fn if is_self_supervised_label(f, file_format)]
    filenamesGt.sort()
    base_filenames = [split_first_subname(image_basename(val)) for val in filenamesGt]

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(file_root), followlinks=True) for f in fn if is_self_supervised_image(f)]
    filenames = [val for val in filenames if split_first_subname(image_basename(val)) in base_filenames]
    filenames.sort()

    len_files = len(filenamesGt)

    print("Filtering empty labels.")
    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(func)(i, filenameGt, filenames, file_root, output_dir, file_format) for i, filenameGt in enumerate(filenamesGt))
        
