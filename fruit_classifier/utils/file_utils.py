import shutil
from tqdm import tqdm
from pathlib import Path


def copytree(src, dst, symlinks=False, ignore=None):
    """
    Copies files from src directory to destination dst. This is a
    workaround for shutils constraints

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

    src_folder = src.parts[-1]

    if src.is_dir() and not dst.is_dir():
        dst.mkdir()

    for s in tqdm(src.glob('*'), desc='Copying file'):

        if s.is_dir():
            d = dst
            for folder in s.parts[s.parts.index(src_folder) + 1:]:
                d = d.joinpath(folder)

            copytree(src=s, dst=d)

        elif s.is_file():
            d = dst.joinpath(s.name)
            shutil.copy2(str(s), str(d))
        else:
            print(f'Item is not a file or directory: {s}')

