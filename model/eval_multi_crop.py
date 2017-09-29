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
flags.DEFINE_string('pretrained_model_path', '', 'the pre-trained model path')

FLAGS = flags.FLAGS

def main():
    datagen = DataGenerator(FLAGS.train_json, FLAGS.train_image_dir, FLAGS.validate_json, FLAGS.validate_image_dir)
    model = ModelFactory(datagen, net='INCEPTION_RESNET_V2', model_dir=FLAGS.model_dir, fine_tune=True, pretrained_path=FLAGS.pretrained_model_path)
    with tf.Session() as session:
        model.eval_multi_crop(session)

if __name__ == '__main__':
    main()
