import unittest
from pathlib import Path
import shutil
from fruit_classifier.utils.file_utils import copytree


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).absolute().parents[2]
        self.src_path = self.root_dir.joinpath('data')
        self.dst_path = self.root_dir.joinpath('test_data_copy')

    def tearDown(self):
        shutil.rmtree(self.dst_path)
        self.assertFalse(self.dst_path.is_dir(),
                         'The copied files was not deleted')

    def test_copytree(self):
        # NOTE: Runs the test twice to test
        #       1. The clean repo
        #       2. If the test passes if files already exists in dst
        for _ in range(2):
            copytree(self.src_path, self.dst_path)

            self.assertTrue(self.dst_path.is_dir(),
                            'The copied files does not exist')

            src_files = self.src_path.glob('**/*')
            dst_files = self.dst_path.glob('**/*')

            for s, d in zip(src_files, dst_files):
                self.assertEqual(s.name, d.name,
                                 msg='Copied files are not matching\n\n' 
                                 f'{s} \n'
                                 f'{d}')


if __name__ == '__main__':
    unittest.main()
