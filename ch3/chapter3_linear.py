import numpy as np
import tensorflow as tf
tf.set_random_seed(456)

# y = wx + b + N(0,e) 생성중..
# ex) 3-2
#가상 데이터 생성
N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N,1)
noise = np.random.normal(scale=noise_scale,size=(N,1))
#y_np의 형태를 (N,)으로 변환
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))

# 그림 그리기 위한 matbplotlib
# from matplotlib import rc
# rc('text', usetex=True)
import matplotlib.pyplot as plt

# Save image of the data distribution
plt.scatter(x_np, y_np)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 1)
plt.title("Toy Linear Regression Data, "
          r"$y = 5x + 2 + N(0, 1)$")
plt.savefig("ex3-2.png")

"3.3.1 텐서플로의 선형회귀"

'''
선형회귀 모델은 간단합니다.
y = wx + b. w와 b는 학습시키고자 하는 가중치. 이 가중치를 tf.Variable 객체로 변환 합니다.
그 다음 L^2 손실을 생성하기 위해 텐서 연산(tensorial operation)을 사용합니다.

L(x,y) = (y-wx-b)^2
밑의 코드는 텐서플로에서 이와 같은 수학적 연산을 구현한 것입니다. 또한 다양한 연산을 그룹화하기 위해 tf.name_scope을 사용하고 학습을 위해 tf.train.AdamOptimizer와 텐서보드 사용을 위해 tf.summary 연산을 추가합니다.
'''
# 선형 회귀 모델 정의하기
#텐서플로 그래프 생성
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N,1)) # 위에서 N = 100
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    # x 는 스칼라이므로 W는 단일 학습 가능 가중치입니다.
    W = tf.Variable(tf.random_normal((1,1))) # <tf.Variable 'Variable:0' shape=(1, 1) dtype=float32_ref>
    b = tf.Variable(tf.random_normal((1,))) # <tf.Variable 'Variable_1:0' shape=(1,) dtype=float32_ref>
with tf.name_scope("prediction"):
    y_pred = tf.matmul(x, W) + b
with tf.name_scope("loss"):
    # l = tf.reduce_sum((y - y_pred)**2) # 실제 알고있는 값과 에측값의 차이. 지도학습이니..
    l = tf.reduce_sum((y - tf.squeeze(y_pred)) ** 2) # 이걸로 해야 경사 하강법이 제대로 됨.. 왜지?
    #y는 1차원인데 y_pred는 1차원이 아니였나?
#학습 연산 추가
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(.001).minimize(l)
    # 위에서 추천한 대로 학습 비율을 .001로 설정.
     # train_op = tf.train.AdamOptimizer(.001).minimize(l) # 위 차이를 최소화.
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l) # 위 loss의 name_scope에 있는 l을 요약해서 합치는듯
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('trainfile2', tf.get_default_graph()) # 로그파일 작성

# 선형 회귀 모델 학습하기
n_steps = 8000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 모델 학습
    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("step %d, loss: %f" % (i, loss))
        train_writer.add_summary(summary, i)


# 텐서보드 실행
# tensorboard --logdir=/tmp/lr-train