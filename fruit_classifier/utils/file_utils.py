import os
import shutil
from tqdm import tqdm
from pathlib import Path


def copytree(src, dst, symlinks=False, ignore=None):
    """
    Copies files from src directory to destination dst

    Parameters
    ----------
    src : Path
        The source directory
    dst : Path
        The destination directory
    symlinks : bool
        Whether to include symlinks
    ignore : callable
        Function which ignores files

    References
    ----------
    https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    """

    src = Path(src)
    dst = Path(dst)

    for item in tqdm(os.listdir(src), desc='Copying files'):
        s = src.joinpath(item)
        d = dst.joinpath(item)

        if s.is_dir() and not d.exists():
            shutil.copytree(str(s), str(d), symlinks, ignore)
        elif not d.exists():
            shutil.copy2(str(s), str(d))
        else:
            pass

    print(os.listdir(src))
    print(os.listdir(dst))


if __name__ == '__main__':
    #Testing the copytree function

    root_dir = Path(__file__).absolute().parents[1]

    src_path = root_dir.joinpath('data_scraping')
    dst_path = root_dir.joinpath('data_scraping_copy')

    if not dst_path.is_dir():
        dst_path.mkdir(parents=True, exist_ok=True)

    copytree(src=src_path, dst=dst_path)

    shutil.rmtree(dst_path)
