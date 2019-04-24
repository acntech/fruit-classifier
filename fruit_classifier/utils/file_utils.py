import os
import shutil
from tqdm import tqdm


def copytree(src, dst, symlinks=False, ignore=None):
    """
    Workaround for shutils weird constraints

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

    src = str(src)
    dst = str(dst)

    for item in tqdm(os.listdir(src), desc='Copying files'):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
