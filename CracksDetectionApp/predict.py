import argparse
import os as os
from pathlib import Path

import cv2
import numpy as np
import sys
import tensorflow as tf

from dataset import cache

import http.server
from http.server import BaseHTTPRequestHandler
import socketserver

PORT = 8080

tf_session = None
tf_predictions = None
tf_x = None
image_size = 128
iterations = 1


def break_image(test_image, size):
    h, w = np.shape(test_image)[0], np.shape(test_image)[1]
    broken_image = []
    h_no = h // size
    w_no = w // size
    h = h_no * size
    w = w_no * size
    for i in range(0, h_no):
        for j in range(0, w_no):
            split = test_image[size * i:size * (i + 1), size * j:size * (j + 1), :]
            broken_image.append(split);

    return broken_image, h, w, h_no, w_no


counter = 0
class PredictImage(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        img_data = self.rfile.read(content_length)  # <--- Gets the data itself

        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, 3)

        broken_image, h, w, h_no, w_no = break_image(img_np, image_size)

        output_image = np.zeros((h_no * image_size, w_no * image_size, 3), dtype=np.uint8)

        x = tf_x
        feed_dict = {x: broken_image}
        predictions_matrices = []
        global iterations

        for i in range(iterations):
            batch_predictions = tf_session.run(tf_predictions, feed_dict=feed_dict)

            matrix_pred = batch_predictions.reshape((h_no, w_no))
            predictions_matrices.append(matrix_pred)

        # Concentrate after this for post processing
        for i in range(0, h_no):
            for j in range(0, w_no):
                numbers = map(lambda matrix: matrix[i, j], predictions_matrices)
                the_sum = sum(numbers)
                keep = 1
                if the_sum != 0:
                    keep = 0
                output_image[image_size * i:image_size * (i + 1), image_size * j:image_size * (j + 1), :] = keep

        cropped_image = img_np[0:h_no * image_size, 0:w_no * image_size, :]
        pred_image = np.multiply(output_image, cropped_image)

        is_success, buffer = cv2.imencode(".jpg", pred_image)

        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Content-Length', len(buffer))
        self.end_headers()

        # print("request is done, sending back", buffer)
        # global counter
        # cv2.imwrite('/Users/stefancosmin/Faculty/IP/crack-detector/CracksDetectionApp/results/' + str(counter) + 'img.jpg', buffer)
        # counter += 1
        self.wfile.write(buffer)


class Dataset_test:
    def __init__(self, in_dir, exts='.jpg', model_number=1):
        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)

        # Input directory.
        self.in_dir = in_dir

        model = None

        print("LOADING DATA TEST")
        if model_number == 1:
            from trainAlg1 import Model
            print("imported Model 1")
        elif model_number == 2:
            from trainAlg2 import Model
            print("imported Model 2")

        model = Model(in_dir)
        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Filenames for all the files in the test-set
        self.filenames = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = model.num_classes

        # If it is a directory.
        if os.path.isdir(in_dir):

            # Get all the valid filenames in the dir
            self.filenames = self._get_filenames_and_paths(in_dir)

        else:
            print("Invalid Directory")
        self.images = self.load_images(self.filenames)

    def _get_filenames_and_paths(self, dir):
        """
        Create and return a list of filenames with matching extensions in the given directory.
        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.
        :return:
            List of filenames. Only filenames. Does not include the directory.
        """

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    path = os.path.join(self.in_dir, filename)
                    filenames.append(os.path.abspath(path))

        return filenames

    def load_images(self, image_paths):
        # Load the images from disk.
        # images = [cv2.resize(cv2.imread(path), (128, 128)) for path in image_paths]
        images = [cv2.imread(path) for path in image_paths]

        # Convert to a numpy array and returns it in the form of [num_images,size,size,channel]
        return np.asarray(images)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Network')
    parser.add_argument('--in_dir', dest='in_dir', type=str, default='images_test')
    parser.add_argument('--meta_file', dest='meta_file', type=str, default='./model_complete.meta')
    parser.add_argument('--CP_dir', dest='chk_point_dir', type=str, default='.')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--start_as_server', type=bool, default=False)
    parser.add_argument('--model_number', type=int, default=1)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--iterations', type=int, default=1)
    return parser.parse_args()


def main(args):
    # File names are saved into a cache file
    args = parse_arguments()
    dataset_test = cache(cache_path='dataset_cache_test.pkl',
                         fn=Dataset_test,
                         in_dir=args.in_dir,
                         model_number=args.model_number)
    test_images = dataset_test.images

    global PORT
    PORT = args.port
    global iterations

    iterations = args.iterations

    try:
        os.stat(args.save_dir)
    except:
        os.mkdir(args.save_dir)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)) as sess:
            # import the model dir
            try:
                file_ = Path(args.meta_file)
                abs_path = file_.resolve()
            except FileNotFoundError:
                sys.exit('Meta File Not found')
            else:
                imported_meta = tf.train.import_meta_graph(args.meta_file)

            if os.path.isdir(args.chk_point_dir):
                imported_meta.restore(sess, tf.train.latest_checkpoint(args.chk_point_dir))
            else:
                sys.exit("Check Point Directory does not exist")

            x = graph.get_operation_by_name("x").outputs[0]
            predictions = graph.get_operation_by_name("predictions").outputs[0]

            global tf_session
            global tf_predictions
            global tf_x

            tf_session = sess
            tf_predictions = predictions
            tf_x = x
            if args.start_as_server:
                with socketserver.TCPServer(("", PORT), PredictImage) as httpd:
                    print("serving at port", PORT)
                    try:
                        httpd.serve_forever()
                    except KeyboardInterrupt:
                        pass
                    httpd.server_close()
                    print("Closing...")
                    return

            global image_size
            # Take one image at a time, pass it through the network and save it
            for counter, image in enumerate(test_images):
                broken_image, h, w, h_no, w_no = break_image(image, image_size)

                output_image = np.zeros((h_no * image_size, w_no * image_size, 3), dtype=np.uint8)

                feed_dict = {x: broken_image}
                batch_predictions = sess.run(predictions, feed_dict=feed_dict)

                matrix_pred = batch_predictions.reshape((h_no, w_no))
                # Concentrate after this for post processing
                for i in range(0, h_no):
                    for j in range(0, w_no):
                        a = matrix_pred[i, j]
                        output_image[image_size * i:image_size * (i + 1), image_size * j:image_size * (j + 1), :] = 1 - a

                cropped_image = image[0:h_no * image_size, 0:w_no * image_size, :]
                pred_image = np.multiply(output_image, cropped_image)

                print("Saved {} Image(s)".format(counter + 1))
                cv2.imwrite(os.path.join(args.save_dir, 'outfile_{}.jpg'.format(counter + 1)), pred_image)


if __name__ == '__main__':
    main(sys.argv)
