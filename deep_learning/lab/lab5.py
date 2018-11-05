import tensorflow as tf

#logistic classifier
#숙제 : tf.decode_txt 사용해보기 , Kaggle 사이트에서 데이터 구해서 해보기.
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]) , name='weight')
b = tf.Variable(tf.random_normal([1]) , name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -1*tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(100):
        cost_val_ = sess.run([cost,train], feed_dict={X:x_data,Y:y_data})
        if step % 200 == 0:
            print(step, cost_val_)

    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})

    print("\nHypothesis: ",h,"\nCorrect (Y): ",c,"\nAccuracy: ",a)