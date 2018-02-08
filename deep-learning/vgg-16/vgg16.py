import tensorflow as tf
import numpy as np
import cPickle
import matplotlib.pyplot as plt

def scalar_summary(name, x):
    tf.summary.scalar(name, x)
    
def image_summary(name, x):
    tf.summary.image(name, x, max_outputs=10)
    
def histogram_summary(name, x):
    tf.summary.histogram(name, x)

class VGG16Model:
    def __init__(self, weights_path):
        self.weights = np.load(weights_path)
        tf.reset_default_graph()
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.activation_fn = tf.nn.relu
        self.use_bias = True

    def conv2d(self, layer, name, n_filters, trainable, k_size=3):
        return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                activation=self.activation_fn, padding=self.conv_padding, name=name, trainable=trainable,
                                kernel_initializer=tf.constant_initializer(self.weights[name+"_W"], dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(self.weights[name+"_b"], dtype=tf.float32),
                                kernel_regularizer=tf.nn.l2_loss,
                                use_bias=self.use_bias)
    def fc(self, layer, name, size, stddev, trainable):
        # print self.weights[name+"_b"]
        # return tf.layers.dense(layer, size, activation=self.activation_fn,
        #                        name=name, trainable=trainable,
        #                        kernel_initializer=tf.constant_initializer(self.weights[name+"_W"], dtype=tf.float32),
        #                        bias_initializer=tf.constant_initializer(self.weights[name+"_b"], dtype=tf.float32),
        #                        use_bias=self.use_bias)
        return tf.layers.dense(layer, size, activation=self.activation_fn,
                                    name=name, trainable=trainable,
                                    kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev),
                                    bias_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev),
                                    kernel_regularizer=tf.nn.l2_loss,
                                    use_bias=self.use_bias)


    def build_network(self, input_tensor, trainable=False):
        
        self.conv1_1 = self.conv2d(input_tensor, 'conv1_1', 64, trainable)
        histogram_summary('conv1_1_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_1/kernel')[0])
        histogram_summary('conv1_1_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_1/bias')[0])
        
        self.conv1_2 = self.conv2d(self.conv1_1, 'conv1_2', 64, trainable)
        histogram_summary('conv1_2_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_2/kernel')[0])
        histogram_summary('conv1_2_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_2/bias')[0])
        
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv2_1 = self.conv2d(self.max_pool1, 'conv2_1', 128, trainable)
        histogram_summary('conv2_1_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2_1/kernel')[0])
        histogram_summary('conv2_1_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2_1/bias')[0])

        self.conv2_2 = self.conv2d(self.conv2_1, 'conv2_2', 128, trainable)
        histogram_summary('conv2_2_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2_2/kernel')[0])
        histogram_summary('conv2_2_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2_2/bias')[0])
        # self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv3_1 = self.conv2d(self.conv2_2, 'conv3_1', 256, trainable)
        histogram_summary('conv3_1_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_1/kernel')[0])
        histogram_summary('conv3_1_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_1/bias')[0])
        
        self.conv3_2 = self.conv2d(self.conv3_1, 'conv3_2', 256, trainable)
        histogram_summary('conv3_2_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_2/kernel')[0])
        histogram_summary('conv3_2_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_2/bias')[0])
        
        self.conv3_3 = self.conv2d(self.conv3_2, 'conv3_3', 256, trainable)
        histogram_summary('conv3_3_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_3/kernel')[0])
        histogram_summary('conv3_3_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3_3/bias')[0])

        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv4_1 = self.conv2d(self.max_pool3, 'conv4_1', 512, trainable)
        histogram_summary('conv4_1_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_1/kernel')[0])
        histogram_summary('conv4_1_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_1/bias')[0])

        self.conv4_2 = self.conv2d(self.conv4_1, 'conv4_2', 512, trainable)
        histogram_summary('conv4_2_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_2/kernel')[0])
        histogram_summary('conv4_2_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_2/bias')[0])

        self.conv4_3 = self.conv2d(self.conv4_2, 'conv4_3', 512, trainable)
        histogram_summary('conv4_3_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_3/kernel')[0])
        histogram_summary('conv4_3_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4_3/bias')[0])

        # self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv5_1 = self.conv2d(self.conv4_3, 'conv5_1', 512, trainable)
        histogram_summary('conv5_1_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_1/kernel')[0])
        histogram_summary('conv5_1_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_1/bias')[0])

        self.conv5_2 = self.conv2d(self.conv5_1, 'conv5_2', 512, trainable)
        histogram_summary('conv5_2_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_2/kernel')[0])
        histogram_summary('conv5_2_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_2/bias')[0])

        self.conv5_3 = self.conv2d(self.conv5_2, 'conv5_3', 512, trainable)
        histogram_summary('conv5_3_W', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_3/kernel')[0])
        histogram_summary('conv5_3_b', tf.get_collection(tf.GraphKeys.VARIABLES, 'conv5_3/bias')[0])

        # self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)

        reshaped = tf.reshape(self.conv5_3, shape=(-1, 8 * 8 * 512))

        self.fc6 = self.fc(reshaped, 'fc6', 2048, 0.005, trainable)
        self.fc7 = self.fc(self.fc6, 'fc7', 2048,  0.022, trainable)

        self.logits = self.fc(self.fc7, 'fc8', 10, 0.022, trainable)

        self.predictions = tf.nn.softmax(self.logits, name='predictions')

    def load_cifar_dataset(self, file_path):
        with open(file_path, 'rb') as fo:
            data = cPickle.load(fo)
        x = data["data"].reshape(-1, 3, 32, 32)  
        x = np.rollaxis(x,1,4) - np.array([123.68, 116.779, 103.939])
        y = data["labels"]

        return x, y

    def get_next_batch(self, batch_size):
        start = self.last_start_idx + 1
        stop = min(start + batch_size, self.train_x.shape[0])
        batch_x = self.train_x[start:stop]
        batch_y = self.train_y[start:stop]

        if stop == self.train_x.shape[0]:
            self.last_start_idx = -1
            self.epoch += 1
        else:
            self.last_start_idx = stop
        
        return batch_x, batch_y

    def get_random_batch(self, x, y, batch_size):
        rand_idx= np.random.choice(np.arange(len(y)), batch_size, replace=False)
        batch_x = x[rand_idx, :, :, :]
        batch_y = np.array(y)[rand_idx]

        return batch_x, batch_y 

    # problem here
    def predict(self, x, y, batch_size, sess):
        num_batches = len(y)/(batch_size)
        splits_x = np.array_split(x, num_batches)
        splits_y = np.array_split(y, num_batches)
        correct = 0
        
        for split_x, split_y in zip(splits_x, splits_y):
            preds = sess.run([self.predictions], feed_dict={self.X: split_x})[0]
            y_preds = np.argmax(preds,axis=1)            
            correct += np.sum(y_preds == split_y)

        accuracy = correct*100.0/len(y)
        return accuracy

    def train(self):
        self.last_start_idx = -1
        self.epoch = 0
        summary_step = 200
        loss_display_step = 200
        learning_rate = 0.01
        max_epochs = 20
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="images")
        self.y = tf.placeholder(tf.int32, shape=(None))
        
        self.train_x, self.train_y = self.load_cifar_dataset("cifar-10-batches-py/data_batch_2")
        self.test_x, self.test_y = self.load_cifar_dataset("cifar-10-batches-py/test_batch")

        self.build_network(self.X, trainable=True)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        
        scalar_summary('loss', loss)
        
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
        train_step = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('summary_dir', sess.graph)
            merged = tf.summary.merge_all()
            sess.run(init)
            while self.epoch < max_epochs:
                train_step += 1
                batch_x, batch_y = self.get_next_batch(64)
                batch_loss, _ = sess.run([loss,train_op], feed_dict={self.X: batch_x , self.y: batch_y})
                # print sess.run(tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_1/bias')[0])
                if train_step % loss_display_step == 0:
                    test_batch_size = 250

                    test_accuracy = self.predict(self.test_x[:1000], self.test_y[:1000], test_batch_size, sess)
                    sample_train_x, sample_train_y = self.get_random_batch(self.train_x, self.train_y, test_batch_size)
                    
                    preds = sess.run([self.predictions], feed_dict={self.X: sample_train_x})[0]
                    y_preds = np.argmax(preds,axis=1)
                    correct = np.sum(y_preds == sample_train_y)
                    train_accuracy = correct*100.0/len(sample_train_y)
                    print "Loss:",batch_loss," Train Step:",train_step," Train Accuracy:",train_accuracy," Test Accuracy",test_accuracy

                if train_step % summary_step == 0:
                    summary = sess.run(merged, feed_dict={self.X: batch_x, self.y: batch_y})
                    summary_writer.add_summary(summary, train_step)

def main():
    vgg16 = VGG16Model(weights_path="vgg16_weights.npz")
    vgg16.train()

if __name__=="__main__":
    main()