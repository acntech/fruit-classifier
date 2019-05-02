import unittest
from pathlib import Path

from fruit_classifier.utils.file_utils import copytree

class TestFileUtils(unittest.TestCase):

    def setUp(self):
        # make test directories


    def test_copytree(self):
        # Copy the tree by calling function

    def tearDown(self):
        # Delete the copied files and original files
        p = Path("potet")
        e = Path("eple")
        with p.open("rb") as f:
            e.write(f.read())

        with open("pussymagnet", "rb") as f:
            with open("grotteost", "wb") as g:
                g.write(f.read())

        f.close()