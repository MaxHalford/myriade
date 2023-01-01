import collections
import contextlib
import io
import math
import os
import pathlib
import shutil
import tarfile
import tempfile
from urllib import request
import zipfile

from scipy import sparse


__all__ = ["get_data_home", "load_dmoz", "load_wiki_small", "load_wiki_large"]


def get_data_home() -> pathlib.Path:
    """Return the location where datasets are to be stored."""

    data_home = os.environ.get("MYRIAD_DATA", os.path.join("~", "myriad_data"))
    data_home = os.path.expanduser(data_home)
    data_home = pathlib.Path(data_home)
    if not data_home.exists():
        os.makedirs(data_home)
    return data_home


def humanize_bytes(n_bytes: int) -> str:
    """Returns a human-friendly byte size."""
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    human = n_bytes
    rank = 0
    if n_bytes != 0:
        rank = int((math.log10(n_bytes)) / 3)
        rank = min(rank, len(suffixes) - 1)
        human = n_bytes / (1024.0**rank)
    f = ("%.2f" % human).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[rank])


def download(url, f, verbose=True):
    """Download the contents of url into f.

    Parameters:
        url: A file URL.
        f: A file-like object.
        verbose: Whether or not to display download information.

    """

    with request.urlopen(url) as r:
        # Notify the user
        if verbose:
            meta = r.info()
            try:
                n_bytes = int(meta["Content-Length"])
                msg = f"Downloading {url} ({humanize_bytes(n_bytes)})"
            except KeyError:
                msg = f"Downloading {url}"
            print(msg)

        # Now dump the contents of the requests
        shutil.copyfileobj(r, f)


def load_multilabel_svm(path, n_samples, n_labels):
    import svmloader

    X, y = svmloader.load_svmfile(path, multilabels=True)

    # y is a list of tuples, so we encode it into a sparse matrix
    encoder = collections.defaultdict(lambda: len(encoder))
    Y = sparse.dok_matrix((n_samples, n_labels), dtype=bool)
    for i, labels in enumerate(y):
        for label in labels:
            j = encoder[label]
            Y[i, j] = True

    return X, Y.tocsc()


def load_dmoz(verbose=True):
    """Load the DMOZ dataset.

    This dataset comes from the second edition of the Large-Scale Text Classification
    Benchmark (LSTCB).

    Parameters:
        verbose: Whether or not to display download information.

    References:
        1. Partalas, I., Kosmopoulos, A., Baskiotis, N., Artieres, T., Paliouras, G., Gaussier, E.,
            Androutsopoulos, I., Amini, M.R. and Galinari, P., 2015. Lshtc: A benchmark for
            large-scale text classification. arXiv preprint arXiv:1503.08581.

    """

    path = get_data_home().joinpath("dmoz.svm")

    # Load the file if everything seems okay
    if path.exists() and path.stat().st_size == 614_783_119:
        return load_multilabel_svm(path, n_samples=394_756, n_labels=36_372)

    # If not, download and tidy, then try again
    with tempfile.NamedTemporaryFile() as tmp:
        download(
            "http://lshtc.iit.demokritos.gr/files/second/datasets/DMOZ.zip",
            tmp,
            verbose,
        )

        with zipfile.ZipFile(tmp.name, "r") as zf, open(path, "w") as out:
            train = zf.open("train.txt", "r")
            train = io.TextIOWrapper(train)

            for line in train:
                # Replace the label separator ', ' by ','
                commas = line.split(":", maxsplit=1)[0].count(",")
                if commas:
                    line = line.replace(", ", ",", commas)

                print(line, file=out, end="")

    return load_dmoz(verbose)


def load_wiki_small(verbose=True):
    """Load the small Wikipedia dataset.

    This dataset comes from the second edition of the Large-Scale Text Classification
    Benchmark (LSTCB).

    Parameters:
        verbose: Whether or not to display download information.

    References:
        1. Partalas, I., Kosmopoulos, A., Baskiotis, N., Artieres, T., Paliouras, G., Gaussier, E.,
            Androutsopoulos, I., Amini, M.R. and Galinari, P., 2015. Lshtc: A benchmark for
            large-scale text classification. arXiv preprint arXiv:1503.08581.

    """

    path = get_data_home().joinpath("wiki_small.svm")

    # Load the file if everything seems okay
    if path.exists() and path.stat().st_size == 210_139_493:
        return load_multilabel_svm(path, n_samples=456_886, n_labels=36_504)

    # If not, download and tidy, then try again
    tar_path = path.with_name("wiki_small.tar")
    with open(tar_path, "wb") as tar:
        url = "http://lshtc.iit.demokritos.gr/files/second/datasets/wikipediaSmallv2.0.tar.gz"
        download(url, tar, verbose)

    with open(tar_path, "rb") as tar:
        # Extract train.txt from the archive
        with tarfile.open(tar_path) as wiki:
            wiki.extract("train.txt", path=path.parent)
        tar_path.unlink()

        with open(path.with_name("train.txt")) as train, open(path, "w") as out:

            def split(pair):
                k, v = pair.split(":")
                return int(k), v

            for line in train:
                # Replace the label separator ', ' by ','
                commas = line.split(":", maxsplit=1)[0].count(",")
                if commas:
                    line = line.replace(", ", ",", commas)

                # Sort the features
                labels, features = line.rstrip().split(" ", 1)
                features = sorted(map(split, (pair for pair in features.split(" "))))
                features = " ".join(f"{k}:{v}" for k, v in features)

                print(f"{labels}  {features}", file=out)

        path.with_name("train.txt").unlink()

    return load_wiki_small(verbose)


def load_wiki_large(verbose=True):
    """Load the large Wikipedia dataset.

    This dataset comes from the second edition of the Large-Scale Text Classification
    Benchmark (LSTCB).

    Parameters:
        verbose: Whether or not to display download information.

    References:
        1. Partalas, I., Kosmopoulos, A., Baskiotis, N., Artieres, T., Paliouras, G., Gaussier, E.,
            Androutsopoulos, I., Amini, M.R. and Galinari, P., 2015. Lshtc: A benchmark for
            large-scale text classification. arXiv preprint arXiv:1503.08581.

    """

    path = get_data_home().joinpath("wiki_large.svm")

    # Load the file if everything seems okay
    if path.exists() and path.stat().st_size == 1_008_404_622:
        return load_multilabel_svm(path, n_samples=2_365_436, n_labels=325_056)

    # If not, download and tidy, then try again
    tar_path = path.with_name("wiki_large.tar")
    with open(tar_path, "wb") as tar:
        url = "http://lshtc.iit.demokritos.gr/files/second/datasets/wikipediaLarge2.0.tar.gz"
        download(url, tar, verbose)

    with open(tar_path, "rb") as tar:
        # Extract train.txt from the archive
        with tarfile.open(tar_path) as wiki:
            wiki.extract("train.txt", path=path.parent)
        tar_path.unlink()

        with open(path.with_name("train.txt")) as train, open(path, "w") as out:

            def split(pair):
                k, v = pair.split(":")
                return int(k), v

            for line in train:
                # Replace the label separator ', ' by ','
                commas = line.split(":", maxsplit=1)[0].count(",")
                if commas:
                    line = line.replace(", ", ",", commas)

                # Sort the features
                labels, features = line.rstrip().split(" ", 1)
                features = sorted(map(split, (pair for pair in features.split(" "))))
                features = " ".join(f"{k}:{v}" for k, v in features)

                print(f"{labels}  {features}", file=out)

        path.with_name("train.txt").unlink()

    return load_wiki_large(verbose)
