import cv2
import numpy as np
import json
import os

class DataGenerator():

    def __init__(self, train_json_file, train_image_dir, validate_json_file, validate_image_dir, test_json_file=None, test_image_dir=None, input_size=224):

        print 'Loading train and validate samples...'
        self.train_samples = self.load_samples(train_json_file)
        self.validate_samples = self.load_samples(validate_json_file)
        self.train_image_dir = train_image_dir
        self.validate_image_dir = validate_image_dir
        self.input_size = input_size

        self.train_count = len(self.train_samples)
        self.validate_count = len(self.validate_samples)

        if test_json_file:
            self.test_samples = self.load_image_samples(test_json_file)
            self.test_image_dir = test_image_dir
            self.test_count = len(self.test_samples)


        print 'train samples {}, validate samples {}'.format(self.train_count, self.validate_count)

    def load_samples(self, json_file):
        samples = []
        j_f = json.load(open(json_file))
        for sample in j_f:
            samples.append({
                'image_id': sample['image_id'],
                'label_id': sample['label_id']
            })
        return samples

    def load_image_samples(self, json_file):
        samples = []
        j_f = json.load(open(json_file))
        for sample in j_f:
            samples.append(sample['image_id'])
        return samples

    def load_image_from_file(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = img / 255.0
        return img

    def generate_batch_train_samples(self, batch_size=32):
        index = 0
        while True:
            batch_x = []
            batch_y = []
            for i in xrange(batch_size):
                sample = self.train_samples[index % self.train_count]
                image = self.load_image_from_file(os.path.join(self.train_image_dir, sample['image_id']))
                label = int(sample['label_id'])
                batch_x.append(image)
                batch_y.append(label)

                index += 1

            batch_x = np.asarray(batch_x)
            batch_y = np.asarray(batch_y)

            yield batch_x, batch_y

    def get_validate_sample_count(self):
        return self.validate_count

    def generate_validate_samples(self, batch_size=32):
        index = 0
        while index < self.validate_count:
            batch_x = []
            batch_y = []
            for i in xrange(batch_size):
                sample = self.validate_samples[index]
                image = self.load_image_from_file(os.path.join(self.validate_image_dir, sample['image_id']))
                label = int(sample['label_id'])
                batch_x.append(image)
                batch_y.append(label)

                index += 1
                if index == self.validate_count:
                    break

            batch_x = np.asarray(batch_x)
            batch_y = np.asarray(batch_y)

            yield batch_x, batch_y

    def get_test_sample_count(self):
        return self.test_count

    def generate_test_samples(self, batch_size=32):
        index = 0
        while index < self.test_count:
            batch_x = []
            for i in xrange(batch_size):
                sample = self.test_samples[index]
                image = self.load_image_from_file(os.path.join(self.validate_image_dir, sample))

                batch_x.append(image)

                index += 1
                if index == self.test_count:
                    break

            batch_x = np.asarray(batch_x)

            yield batch_x
