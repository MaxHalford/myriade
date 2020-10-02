# myriad

Multiclass classification with tens of thousands of classes

## Usage

## Datasets

| Name | Function    | Size     | Samples | Features | Labels     | Multi-label    | Labels/sample |
|:----:|:-----------:|:---------|:-------:|:--------:|:----------:|:--------------:|:-------------:|
| DMOZ | `load_dmoz` | 614,8 MB | 394,756 | 833,484  | 36,372     | ✓              | 1.02          |
| Wikipedia (small) | `load_wiki_small` | 135,5 MB | 456,886 | 2,085,165  | 36,504     | ✓              | 1.84          |
| Wikipedia (large) | `load_wiki_large` | 1,01 GB | 2,365,436 | 2,085,167  | 325,056     | ✓              | 3.26          |

Each `load_*` function returns two arrays which contain the features and the target classes, respectively. In the multi-label case, the target array is 2D. The arrays are sparse when applicable.

```py
>>> from myriad import datasets

>>> X, y = datasets.load_dmoz()
>>> X

>>> y

```

The first time you call a `load_*` function, the data will be downloaded and saved into a `.svm` file that adheres to the [LIBSVM format convention](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q03:_Data_preparation). The loaders will restart from scratch if you interrupt them during their work.

All of the datasets are loaded in memory with the [`svmloader`](https://github.com/yoch/svmloader/) library. The latter is much faster than the [`load_svmlight_file`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html) function from scikit-learn. However, when working repeatedly on the same dataset, it is recommended to wrap the dataset loader with [`joblib.Memory.cache`](https://joblib.readthedocs.io/en/latest/memory.html) to store a memmapped backup of the results of the first call. This enables near instantaneous loading for subsequent calls.

You can see where the datasets are stored as so:

```py
>>> datasets.get_data_home()

```

## Benchmarks
