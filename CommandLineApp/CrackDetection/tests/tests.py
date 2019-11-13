import unittest
import os

import ImageProcesser

class TestStringMethods(unittest.TestCase):

    def test_folder_NOK(self):
        exists, directorName = ImageProcesser.getImagesFolder("")
        self.assertFalse(exists)
        print("------------------  " + directorName)

    def test_folder_OK(self):
        exists, directorName = ImageProcesser.getImagesFolder(os.path.abspath(__file__ + "/../../../"))
        self.assertTrue(exists)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
