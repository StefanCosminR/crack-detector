from unittest import TestCase

from unittest import TestCase
import CracksDetectionApp.trainAlg2 as Alg2
import tensorflow as tf


class TestBasics(TestCase):
    model = Alg2.Model("")

    def test_new_conv_layer(self):
        layer = Alg2.new_conv_layer(self.model.x_image, self.model.num_channels, 10, 10)
        self.assertIsNotNone(layer)

    def test_max_pool(self):
        layer = Alg2.max_pool(self.model.x_image, [1, 7, 7, 1], [1, 2, 2, 1])
        self.assertIsNotNone(layer)

    def test_new_fc_layer(self):
        layer_flat, num_features = Alg2.flatten_layer(self.model.x_image)
        layer = Alg2.new_fc_layer(layer_flat, num_features, 96)
        self.assertIsNotNone(layer)

    def test_flatten_layer(self):
        layer = Alg2.flatten_layer(self.model.x_image)
        self.assertIsNotNone(layer)


class TestModel(TestCase):
    model = Alg2.Model("")

    def test_load_images(self):
        images = self.model.load_images("")
        self.assertEqual(images.size, 0)

    def test_define_model(self):
        optimizer, accuracy = self.model.define_model()
        self.assertIsNotNone(optimizer)
        # self.assertLessEqual(accuracy, 100)

    def test_random_batch(self):
        x_batch, y_batch = self.model.random_batch()
        self.assertIsNotNone(x_batch)
        self.assertIsNotNone(y_batch)

    def test_print_test_accuracy(self):
        with tf.Session() as sess:
            self.model.print_test_accuracy(sess)
            self.assertTrue(True, "Pass")

    def test_optimize(self):
        self.model.optimize(0)
        self.assertEqual(self.model.total_iterations, 0)
        self.assertTrue(self.model.time_dif > 0)

    def test_parse_arguments(self):
        self.assertIsNotNone(self)
