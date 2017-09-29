import cv2
import numpy as np
import json
import os
import random as rd

class DataGenerator():

    def __init__(self, train_json_file, train_image_dir, validate_json_file, validate_image_dir, test_json_file=None, test_image_dir=None, input_size=299, data_augmentation=True):

        print 'Loading train and validate samples...'
        self.train_samples = self.load_samples(train_json_file)
        self.validate_samples = self.load_samples(validate_json_file)
        self.train_image_dir = train_image_dir
        self.validate_image_dir = validate_image_dir
        self.input_size = input_size

        self.train_count = len(self.train_samples)
        self.validate_count = len(self.validate_samples)
        self.data_augmentation = data_augmentation

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
        return img

    def generate_batch_train_samples(self, batch_size=32):
        index = 0
        batch_x = np.zeros((batch_size, self.input_size, self.input_size, 3))
        batch_y = np.zeros(batch_size)
        while True:
            for i in xrange(batch_size):
                sample = self.train_samples[index]
                image = self.load_image_from_file(os.path.join(self.train_image_dir, sample['image_id']))

                if self.data_augmentation:
                    image = self.image_aug(image)

                label = int(sample['label_id'])
                batch_x[i] = image
                batch_y[i] = label

                index += 1
                index = index % self.train_count

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
                image = cv2.resize(image, (self.input_size, self.input_size))
                label = int(sample['label_id'])
                batch_x.append(image)
                batch_y.append(label)

                index += 1
                if index == self.validate_count:
                    break

            batch_x = np.asarray(batch_x)
            batch_y = np.asarray(batch_y)

            yield batch_x, batch_y

    def generate_validate_samples_Y(self):
        batch_y = []
        for index in range(self.validate_count):
            sample = self.validate_samples[index]
            label = int(sample['label_id'])
            batch_y.append(label)

        batch_y = np.asarray(batch_y)
        return batch_y

    def generate_scaled_validate_samples_X(self, resize=299, batch_size=32):
        index = 0
        while index < self.validate_count:
            batch_x = []
            for i in xrange(batch_size):
                sample = self.validate_samples[index]
                image = self.load_image_from_file(os.path.join(self.validate_image_dir, sample['image_id']))
                image = cv2.resize(image, (resize, resize))
                image = self.crop_center(image, self.input_size, self.input_size)
                batch_x.append(image)

                index += 1
                if index == self.validate_count:
                    break

            batch_x = np.asarray(batch_x)

            yield batch_x

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

    def image_aug(self, image):
        crop_size = rd.randint(300, 350)
        image = cv2.resize(image, (crop_size, crop_size))
        image = self.random_crop(image, self.input_size, self.input_size)

        # horizental flip
        flip = rd.randint(0, 1)
        if flip == 1:
            image = cv2.flip(image, 1)
        # brightless
        bright = rd.randint(0, 1)
        if bright == 1:
            scale = rd.uniform(0.8, 1.2)
            image = image * scale

        return image

    def random_crop(self, img, w, h):
        h_total = img.shape[0]
        w_total = img.shape[1]

        left = rd.randint(0, w_total - w)
        right = left + w
        top = rd.randint(0, h_total - h)
        bot = top + h
        return img[top:bot, left:right, :].copy()

    def crop_center(self, img, w, h):
        h_total = img.shape[0]
        w_total = img.shape[1]

        left = w_total / 2 - w / 2
        right = left + w
        top = h_total / 2 - h / 2
        bot = top + h
        return img[top:bot, left:right, :].copy()

    def resize_shorter(self, img, size):
        h = img.shape[0]
        w = img.shape[1]

        if h > w:
            r_w = size
            r_h = h * r_w / w
        else:
            r_h = size
            r_w = w * r_h / h

        return cv2.resize(img, (r_w, r_h))


