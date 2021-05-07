import os
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import argparse


def _image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image):

    feature = {
        "image": _image_feature(image)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def create_tfrecord(tfrecord_fname, image_dir):

    with tf.io.TFRecordWriter(tfrecord_fname) as writer:

        for image_fname in glob.glob('{}/*.jpg'.format(image_dir)):

            image = tf.io.decode_jpeg(tf.io.read_file(image_fname))
            example = create_example(image)
            writer.write(example.SerializeToString())


def parse_tfrecord(example):

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, feature_description)

    # decode jpeg
    image = tf.io.decode_jpeg(example["image"], channels=3)

    return image


def visualize_tfrecord(tfrecord_fname, image_num):

    raw_dataset = tf.data.TFRecordDataset(tfrecord_fname)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    for image in parsed_dataset.take(image_num):

        plt.figure(figsize=(7, 7))
        plt.imshow(image.numpy())
        plt.show()


if __name__ == '__main__':

    # python create_tfrecord.py --category trainA
    # python create_tfrecord.py --category testB

    parser = argparse.ArgumentParser('Description - Create TFRecord')
    parser.add_argument('--category', help='Name of category, e.g., trainA, testB, etc...')
    args = parser.parse_args()

    category = args.category

    dataset_dir = os.path.join('data', 'horse2zebra')
    image_dir = os.path.join(dataset_dir, category)
    tfrecord_fname = os.path.join(dataset_dir, '{}.tfrecord'.format(category))

    # create tfrecord
    # create_tfrecord(tfrecord_fname=tfrecord_fname, image_dir=image_dir)

    # visualize tfrecord
    visualize_tfrecord(tfrecord_fname=tfrecord_fname, image_num=5)