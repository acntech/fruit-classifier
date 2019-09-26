import shutil
from tqdm import tqdm
from pathlib import Path


def copytree(src, dst):
    """
    Copies files from src directory to destination dst.

    This is a workaround for shutils constraints

    Parameters
    ----------
    src : Path
        The source directory
    dst : Path
        The destination directory

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


def truncate_filenames(raw_dir):
    """
    Truncate file names so path length <= 255 characters

    Assumes raw_dir is a directory that contains ONLY directories,
    in which all files/directories will be truncated if their total
    path length is more than 255 characters.

    Parameters
    ----------
    raw_dir: Will truncate files in raw_dir's sub directories
    """

    subdirectory_list = [p for p in Path(raw_dir).glob('*')
                         if p.is_dir()]
    print("Reducing length of filenames so that combined path to a "
          "file is maximum 255 characters long")
    max_windows_path_length = 255

    for sub_dir_path in subdirectory_list:

        # length of directory path + "/"
        sub_dir_len = len(str(sub_dir_path)) + 1
        # length available for image file name including type
        available_max_len = max_windows_path_length - sub_dir_len
        # Get all files in directory
        file_list = list(Path(sub_dir_path).glob('*'))
        num_renamed = 0
        for filepath in file_list:
            filename = filepath.name
            file_path_len = len(filename)
            if file_path_len <= max_windows_path_length:
                continue
            possible_types = filename.split('.')
            file_type = possible_types[-1]
            cut_position = available_max_len - len(file_type) - 1
            new_name = filename[0:cut_position] + '.' + file_type

            old_path = sub_dir_path.joinpath(filename)
            new_path = sub_dir_path.joinpath(new_name)

            Path.rename(old_path, new_path)
            num_renamed = num_renamed + 1
        print(f'[INFO] Truncated {num_renamed} filenames in directory: '
              f'{sub_dir_path.name}')
