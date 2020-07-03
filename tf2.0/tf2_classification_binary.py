import tensorflow as tf

x_data = tf.Variable([[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]],dtype=tf.float32,shape=[None,2])
y_data = tf.Variable([[0], [0], [0], [1], [1], [1]],dtype=tf.float32)

W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

def compute_cost():
    model = tf.sigmoid(tf.matmul(x_data,W)+b)
    cost = tf.reduce_mean((-1)*y_data*tf.math.log(model) + (-1)*(1-y_data)*tf.       math.log(1-model))
    return cost


train = tf.keras.optimizers.SGD(lr=0.01)

prediction = tf.cast(model > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data), dtype=tf.float32))

for step in range(10001):
    train.minimize(compute_cost,var_list=[W,b])
    print(step, '   cost:',compute_cost().numpy())

#test
h = tf.sigmoid(tf.matmul(x_data,W)+b)
c = tf.cast(h > 0.5, dtype=tf.float32)
a = tf.reduce_mean(tf.cast(tf.equal(c,y_data),dtype=tf.float32))
print("\nModel: ",h .numpy(), "\nCorrect: ", c.numpy(), "\nAccuracy: ", a.numpy())