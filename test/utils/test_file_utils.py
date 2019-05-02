import unittest
from pathlib import Path
import shutil
from fruit_classifier.utils.file_utils import copytree
import os


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).absolute().parents[1]
        self.src_path = self.root_dir.joinpath('test_data')
        self.dst_path = self.root_dir.joinpath('test_data_copy')

        if not self.dst_path.is_dir():
            self.dst_path.mkdir(parents=True, exist_ok=True)

    def test_copytree(self):
        self.compare_folders()
        self.compare_folders()

    def compare_folders(self):
        copytree(self.src_path, self.dst_path)

        self.assertTrue(self.dst_path.is_dir(), 'The copied files does not exist')

        src_files = os.listdir(self.src_path)
        dst_files = os.listdir(self.dst_path)

        for i, item in enumerate(src_files):
            self.assertEqual(dst_files[i], item,
                            msg='Copied files are not matching\n\n ' 
                            f'{dst_files[i]} \n'
                            f'{item}')

    def tearDown(self):
        shutil.rmtree(self.dst_path)
        self.assertFalse(self.dst_path.is_dir(), 'The copied files was not deleted')


if __name__ == '__main__':
    unittest.main()