import unittest
import tensorflow as tf
import argparse

from CracksDetectionApp.src import trainAlg1


class trainAlg1Test(tf.test.TestCase):

    def testArgumentList(self):
        # setup
        expected_args = self.createArgsList()

        # execute
        returned_args = trainAlg1.parse_arguments()

        # verify
        self.assertEqual(expected_args, returned_args)

    def testLoadImages(self):
        # setup
        args = self.createArgsList()

        # execute
        model = trainAlg1.Model(args.in_dir, args.save_folder)

        # verify
        self.assertTrue(model)

    def test_model_rights(self):
        # verify
        self.assertFalse(trainAlg1.checkModelWriteRights(""))

    def createArgsList(self):
        expected_args = argparse.Namespace()
        expected_args.script = 'train_alg1_test.py:trainAlg1Test.testArgumentList'
        expected_args.in_dir = 'image__big_set'
        expected_args.num_iterations = 15000
        expected_args.save_folder = 'alg1_output'
        return expected_args


if __name__ == '__main__':
    unittest.main()
