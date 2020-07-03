import tensorflow as tf

x_data = tf.Variable([[1,1], [2,2], [3,3]],dtype=tf.float32)
y_data = [[10], [20], [30]]

W=tf.Variable(tf.random.normal([2,1]))
b=tf.Variable(tf.random.normal([1]))

def compute_cost():
    model = tf.matmul(x_data, W) + b
    cost = tf.reduce_mean((model - y_data)**2)
    return cost

train = tf.keras.optimizers.SGD(lr=0.01)
for step in range(2001):
    train.minimize(compute_cost, var_list=[W,b])
    print(step,'cost:',compute_cost().numpy(),'W:',W.numpy(),'b:',b.numpy())
    
#test
predict = tf.matmul(tf.Variable([[4,4]],dtype=tf.float32), W) + b
print(predict.numpy())