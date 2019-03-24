import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a,b)

sess = tf.Session()

file_writer = tf.summary.FileWriter('logfiles', sess.graph)
# (tensorflow)$ tensorboard --logdir=<trace file>
print (sess.run(y, feed_dict={a:3,b:3}))
