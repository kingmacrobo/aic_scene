import tensorflow as tf
import os
import json
import time
from nets import vgg

net_dict = {
    'VGG16': vgg.vgg_16,
}

num_class = 80
slim = tf.contrib.slim

class ModelFactory():
    def __init__(self, datagen, net='VGG16', batch_size=64, lr=0.001, dropout_keep_prob=0.5, model_dir='checkpoints', input_size=224, fine_tune=False, pretrained_path=None):

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
        with slim.arg_scope(vgg.vgg_arg_scope()):
            train_net, _ = self.net(x, num_classes=num_class, dropout_keep_prob=self.dropout_keep_prob)

        # softmax cross entropy loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=train_net))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(self.lr, global_step,
                                                   30000, 0.95, staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss,
            global_step=global_step
        )

        saver = tf.train.Saver(max_to_keep=3)

        # evaluate net
        tf.get_variable_scope().reuse_variables()
        eval_x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        with slim.arg_scope(vgg.vgg_arg_scope()):
            eval_net, _ = self.net(eval_x, num_classes=num_class, dropout_keep_prob=self.dropout_keep_prob, is_training=False)
        _, top_3 = tf.nn.top_k(eval_net, k=3)

        # load vgg pre-trained parameters on ImageNet
        init_fn=None
        fc8_init=None
        if self.fine_tune and not os.path.exists(self.model_dir):
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                print 'Load pretrained model from {}'.format(self.pretrained_path)
                variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.pretrained_path, variables_to_restore)
                # init fc8 layer parameters
                fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
                fc8_init = tf.variables_initializer(fc8_variables)

        if init_fn is not None:
            init_fn(session)
            session.run(fc8_init)

        else:
            session.run(tf.global_variables_initializer())

        # restore the model
        last_step = -1
        last_acc = 0
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            if os.path.exists(self.acc_file):
                acc_json = json.load(open(self.acc_file, 'r'))
                last_acc = acc_json['accuracy']
                last_step = acc_json['step']
            print 'Model restored from {}, last accuracy: {}, last step: {}'\
                .format(ckpt.model_checkpoint_path, last_acc, last_step)


        generate_train_batch = self.datagen.generate_batch_train_samples(batch_size=self.batch_size)
        total_loss = 0
        count = 0
        for step in xrange(last_step + 1, 100000000):
            gd_a = time.time()
            batch_x, batch_y = generate_train_batch.next()
            gd_b = time.time()

            tr_a = time.time()
            _, loss_out = session.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
            tr_b = time.time()

            total_loss += loss_out
            count += 1

            if step % 50 == 0:
                avg_loss = total_loss/count
                print 'global step {}, epoch {}, step {}, loss {}, generate data time: {:.2f} s, step train time: {:.2f} s'\
                    .format(step, step / 53879, step % 53879, avg_loss, gd_b - gd_a, tr_b - tr_a)
                self.loss_log.write('{} {}\n'.format(step, avg_loss))
                total_loss = 0
                count = 0

            if step != 0 and step % 1000 == 0:
                model_path = saver.save(session, os.path.join(self.model_dir, self.net_name))
                if os.path.exists(self.acc_file):
                    j_dict = json.load(open(self.acc_file))
                else:
                    j_dict = {'accuracy': 0}

                j_dict['step'] = step
                json.dump(j_dict, open(self.acc_file, 'w'), indent=4)
                print 'Save model at {}'.format(model_path)

            if step != 0 and step % 5000 == 0:
                print 'Evaluate validate set ... '
                ee_a = time.time()
                correct = 0
                N = self.datagen.get_validate_sample_count()
                batches = N / self.batch_size
                if N % self.batch_size != 0:
                    batches += 1
                validate_samples = self.datagen.generate_validate_samples(self.batch_size)
                for i in xrange(batches):
                    val_x, val_y = validate_samples.next()

                    val_top_3 = session.run(top_3, feed_dict={eval_x: val_x})

                    for j, row in enumerate(val_top_3):
                        if val_y[j] in row:
                            correct += 1

                ee_b = time.time()
                top_3_acc = correct * 1.0 / N

                print 'validate accuracy: {:.5f}, time: {:.2f} s' \
                    .format(top_3_acc, ee_b - ee_a)

                self.acc_log.write('{} {}\n'.format(step, top_3_acc))

                # save model if get higher accuracy
                if top_3_acc > last_acc:
                    last_acc = top_3_acc
                    model_path = saver.save(session, os.path.join(self.model_dir, self.net_name + '_best'))
                    acc_json = {'accuracy': last_acc, 'step': step}
                    with open(self.acc_file, 'w') as f:
                        json.dump(acc_json, f, indent=4)

                    print 'Get higher accuracy, {}. Save model at {}, Save accuracy at {}'\
                        .format(last_acc, model_path, self.acc_file)
