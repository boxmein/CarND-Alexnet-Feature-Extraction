import time
import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open("/input/train.p", "rb") as f: 
	data = pickle.load(f)
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data["features"], data["labels"], test_size=0.33)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)

resized = tf.image.resize_images(x, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fx_W = tf.Variable( tf.truncated_normal( (fc7.get_shape().as_list()[-1], 43)) )
fx_b = tf.Variable( tf.zeros(43) )

logits = tf.matmul(fc7, fx_W) + fx_b 
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_op = tf.reduce_mean(cross_entropy)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opti.minimize(loss_op)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
predictions = tf.arg_max(logits, 1)

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

epochs = 10
batch_size = 512

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("EPOCH\tACC\tLOSS\tTIME")
	for i in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		start = time.time()
		for offset in range(0, X_train.shape[0], batch_size):
			offset_end = offset + batch_size
			sess.run(train_op, feed_dict={
				x: X_train[offset:offset_end],
				y: y_train[offset:offset_end]
			})
		v_loss, v_acc = eval_on_data(X_valid, y_valid, sess)
		end = time.time()
		print("{}\t{:.3f}\t{:.3f}\t{:.3f}".format({
			i, v_loss, v_acc, end - start 
		}))
