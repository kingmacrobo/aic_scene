import tensorflow as tf
import os
import json
import time
import numpy as np
from nets import vgg, inception_resnet_v2, inception_v3

net_dict = {
    'VGG16': vgg.vgg_16,
    'INCEPTION_RESNET_V2': inception_resnet_v2.inception_resnet_v2,
    'INCEPTION_V3': inception_v3.inception_v3,
}

arg_scope_dict = {
    'VGG16': vgg.vgg_arg_scope,
    'INCEPTION_RESNET_V2': inception_resnet_v2.inception_resnet_v2_arg_scope,
    'INCEPTION_V3': inception_v3.inception_v3_arg_scope,
}

num_class = 80
slim = tf.contrib.slim

class ModelFactory():
    def __init__(self, datagen, net='VGG16', batch_size=32, lr=0.001, dropout_keep_prob=0.8, model_dir='checkpoints', input_size=299, fine_tune=False, pretrained_path=None):

        self.datagen = datagen
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_keep_prob = dropout_keep_prob
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.pretrained_path = pretrained_path
        self.acc_file = os.path.join(self.model_dir, 'accuracy.json')
        self.loss_log = open('loss_log', 'w')
        self.acc_log = open('acc_log', 'w')

        self.net_name = net
        self.net = net_dict[net]
        self.input_size = input_size

        print 'Fine-tune model: {}, batch size: {}, learning reate: {}, dropout keep probability: {}\n'.format(self.fine_tune, self.batch_size, self.lr, self.dropout_keep_prob)


    def train(self, session):
        # train net
        x = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])
        y = tf.placeholder(tf.int32, [self.batch_size])
        scaled_x = tf.scalar_mul((1.0 / 255), x)
        scaled_x = tf.subtract(scaled_x, 0.5)
        scaled_x = tf.multiply(scaled_x, 2.0)

        with slim.arg_scope(arg_scope_dict[self.net_name]()):
            train_net, _ = self.net(scaled_x, num_classes=num_class, dropout_keep_prob=self.dropout_keep_prob, reuse=None)

        # load vgg pre-trained parameters on ImageNet
        init_fn=None
        if self.fine_tune and not os.path.exists(self.model_dir):
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                print 'Load pretrained model from {}'.format(self.pretrained_path)
                if self.net_name == 'VGG16':
                    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
                elif self.net_name == 'INCEPTION_RESNET_V2':
                    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits/Logits'])
                elif self.net_name == 'INCEPTION_V3':
                    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['InceptionV3/Logits/Conv2d_1c_1x1', 'InceptionV3/AuxLogits/Conv2d_2b_1x1'])

                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.pretrained_path, variables_to_restore)
                print 'Load pretrained parameters done!'

        # softmax cross entropy loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=train_net))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(self.lr, global_step,
                                                   100000, 0.95, staircase=True)

        last_layer = tf.contrib.framework.get_variables('InceptionResnetV2/Logits')

        '''
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
            loss,
            global_step=global_step
        )
        '''

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9,
            name='Momentum')

        '''
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss,
            global_step=global_step)
        '''

        train_step = slim.learning.create_train_op(loss, optimizer, global_step=global_step, variables_to_train=last_layer)

        saver = tf.train.Saver([v for v in tf.model_variables() if not ('Momentum' in v.name)], max_to_keep=3)

        # evaluate net
        if self.net_name == 'VGG16':
            tf.get_variable_scope().reuse_variables()

        eval_x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        eval_scaled_x = tf.scalar_mul((1.0/255), eval_x)
        eval_scaled_x = tf.subtract(eval_scaled_x, 0.5)
        eval_scaled_x = tf.multiply(eval_scaled_x, 2.0)

        with slim.arg_scope(arg_scope_dict[self.net_name]()):
            eval_net, _ = self.net(eval_scaled_x, num_classes=num_class, dropout_keep_prob=self.dropout_keep_prob, is_training=False, reuse=True)

        eval_net = tf.nn.softmax(eval_net)
        _, top_3 = tf.nn.top_k(eval_net, k=3)
        top_1 = tf.argmax(eval_net, axis=1)

        print 'Initialize uninitialized variables...'
        if init_fn is not None:
            init_fn(session)
            uninit_names = session.run(tf.report_uninitialized_variables())
            print 'Uninitialized variable count: {}'.format(len(uninit_names))
            vis = []
            for v in uninit_names:
                #print 'Initialize {}'.format(v)
                vi = tf.contrib.framework.get_variables(v)
                vis.append(vi[0])
            session.run(tf.variables_initializer(vis))
        else:
            session.run(tf.global_variables_initializer())
        print 'Initialize variables done!'

        # restore the model
        last_step = -1
        last_acc = 0
        if os.path.exists(self.model_dir):
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                if os.path.exists(self.acc_file):
                    acc_json = json.load(open(self.acc_file, 'r'))
                    last_acc = acc_json['accuracy']
                    last_step = acc_json['step']
                print 'Model restored from {}, last accuracy: {}, last step: {}'\
                    .format(ckpt.model_checkpoint_path, last_acc, last_step)


        tf.get_default_graph().finalize()

        generate_train_batch = self.datagen.generate_batch_train_samples(batch_size=self.batch_size)
        total_loss = 0
        count = 0

        print 'Start training...'
        for step in xrange(last_step + 1, 100000000):
            gd_a = time.time()
            batch_x, batch_y = generate_train_batch.next()
            gd_b = time.time()

            tr_a = time.time()
            _, loss_out, c_lr = session.run([train_step, loss, learning_rate], feed_dict={x: batch_x, y: batch_y})
            tr_b = time.time()

            total_loss += loss_out
            count += 1

            if step % 100 == 0:
                avg_loss = total_loss/count
                print 'global step {}, epoch {}, step {}, loss {}, generate data time: {:.2f} s, step train time: {:.2f} s, lr: {}'\
                    .format(step, step / (53879 / self.batch_size), step % (53879 / self.batch_size), avg_loss, gd_b - gd_a, tr_b - tr_a, c_lr)
                self.loss_log.write('{} {}\n'.format(step, avg_loss))
                total_loss = 0
                count = 0

            if step != 0 and step % 1000 == 0:
                if not os.path.exists(self.model_dir):
                    os.mkdir(self.model_dir)
                model_path = saver.save(session, os.path.join(self.model_dir, self.net_name))
                if os.path.exists(self.acc_file):
                    j_dict = json.load(open(self.acc_file))
                else:
                    j_dict = {'accuracy': 0}

                j_dict['step'] = step
                json.dump(j_dict, open(self.acc_file, 'w'), indent=4)
                print 'Save model at {}'.format(model_path)

            if step != 0 and step % 2000 == 0:
                print 'Evaluate validate set ... '
                ee_a = time.time()
                correct = 0
                top_1_correct = 0
                N = self.datagen.get_validate_sample_count()
                batches = N / self.batch_size
                if N % self.batch_size != 0:
                    batches += 1
                validate_samples = self.datagen.generate_validate_samples(self.batch_size)
                for i in xrange(batches):
                    val_x, val_y = validate_samples.next()

                    val_top_3, val_top_1 = session.run([top_3, top_1], feed_dict={eval_x: val_x})

                    for j, row in enumerate(val_top_3):
                        if val_y[j] in row:
                            correct += 1

                    for j, cla in enumerate(val_top_1):
                        if val_y[j] == val_top_1[j]:
                            top_1_correct += 1

                ee_b = time.time()
                top_3_acc = correct * 1.0 / N
                top_1_acc = top_1_correct * 1.0 / N

                print '\nvalidate top-3 acc: {:.5f}, top-1 acc: {},time: {:.2f} s' \
                    .format(top_3_acc, top_1_acc, ee_b - ee_a)

                self.acc_log.write('{} {}\n'.format(step, top_3_acc))

                # save model if get higher accuracy
                if top_3_acc > last_acc:
                    last_acc = top_3_acc
                    if not os.path.exists(self.model_dir):
                        os.mkdir(self.model_dir)
                    model_path = saver.save(session, os.path.join(self.model_dir, self.net_name + '_best'))
                    acc_json = {'accuracy': last_acc, 'step': step}
                    with open(self.acc_file, 'w') as f:
                        json.dump(acc_json, f, indent=4)

                    print '***Get higher accuracy, {}. Save model at {}, Save accuracy at {}\n'\
                        .format(last_acc, model_path, self.acc_file)

    def eval(self, session):
        eval_x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        eval_scaled_x = tf.scalar_mul((1.0/255), eval_x)
        eval_scaled_x = tf.subtract(eval_scaled_x, 0.5)
        eval_scaled_x = tf.multiply(eval_scaled_x, 2.0)

        with slim.arg_scope(arg_scope_dict[self.net_name](weight_decay=0.0)):
            eval_net, _ = self.net(eval_scaled_x, num_classes=num_class, dropout_keep_prob=1, is_training=False, reuse=None)

        eval_net = tf.nn.softmax(eval_net)
        _, top_3 = tf.nn.top_k(eval_net, k=3)
        top_1 = tf.argmax(eval_net, axis=1)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restore the best model
        last_step = -1
        last_acc = 0
        if os.path.exists(self.model_dir):
            ckpt = self.model_dir + '/' + self.net_name + '_best'
            saver.restore(session, ckpt)
            if os.path.exists(self.acc_file):
                acc_json = json.load(open(self.acc_file, 'r'))
                last_acc = acc_json['accuracy']
                last_step = acc_json['step']
            print 'Model restored from {}, last accuracy: {}, last step: {}' \
                .format(ckpt, last_acc, last_step)
        else:
            print 'No model dir'
            return

        print 'Evaluate validate set ... '
        ee_a = time.time()
        correct = 0
        top_1_correct = 0
        N = self.datagen.get_validate_sample_count()
        batches = N / self.batch_size
        if N % self.batch_size != 0:
            batches += 1
        validate_samples = self.datagen.generate_validate_samples(self.batch_size)
        for i in xrange(batches):
            val_x, val_y = validate_samples.next()

            val_top_3, val_top_1 = session.run([top_3, top_1], feed_dict={eval_x: val_x})

            for j, row in enumerate(val_top_3):
                if val_y[j] in row:
                    correct += 1

            for j, cla in enumerate(val_top_1):
                if val_y[j] == val_top_1[j]:
                    top_1_correct += 1

        ee_b = time.time()
        top_3_acc = correct * 1.0 / N
        top_1_acc = top_1_correct * 1.0 / N

        print 'validate top-3 acc: {:.5f}, top-1 acc: {},time: {:.2f} s' \
            .format(top_3_acc, top_1_acc, ee_b - ee_a)
