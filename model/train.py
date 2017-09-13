import tensorflow as tf
from data import DataGenerator
from model_factory import ModelFactory

flags = tf.app.flags

flags.DEFINE_string('model_dir', 'checkpoints', 'the checkpoints directory')
flags.DEFINE_string('train_json', '', 'the file contains train list')
flags.DEFINE_string('validate_json', '', 'the file contains validate list')
flags.DEFINE_string('test_json', '', 'the file contains test list')
flags.DEFINE_string('train_image_dir', '', 'train image directory')
flags.DEFINE_string('validate_image_dir', '', 'validate image directory')
flags.DEFINE_string('test_image_dir', '', 'test image directory')

FLAGS = flags.FLAGS

def main():
    datagen = DataGenerator(FLAGS.train_json, FLAGS.train_image_dir, FLAGS.validate_json, FLAGS.validate_image_dir)
    model = ModelFactory(datagen, net='VGG16', model_dir=FLAGS.model_dir)
    with tf.Session() as session:
        model.train(session)

if __name__ == '__main__':
    main()
