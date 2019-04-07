import numpy as np
np.random.seed(456)
import tensorflow as tf
tf.set_random_seed(456)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import logit

# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
# 0은 (-1, -1)을 중심으로 가우스를 생성합니다.
# 엡실론은 .1입니다.
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))
# Ones form a Gaussian centered at (1, 1)
# 1은 (1,1)을 중심으로 가우스를 생성합니다.
# 엡실론은 .1입니다.
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))

'''
위 가상 데이터셋을 만들기 위한 넘파이 코드는 선형회귀 문제의 경우보다 조금 더 까다롭습니다. 왜냐하면 두 가지 형태의 데이터포인트를 결합하고 서로 다른 레이블과 연결하기 위해 스태킹 함수 np.vstack을 
사용해야 하기 대문입니다(1차원 레이블을 결합하기 위해 관련 함수 np.concatenate를 사용합니다.
'''
x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Save image of the data distribution
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Toy Logistic Regression Data")

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
plt.savefig("logistic_data.png")

"3.3.2 텐서플로의 로지스틱 회귀"

'''
분류기를 위한 방정식은 수학적 기법은 시그모디으 함수(sigmoid function) 이라는게 있다. 실수 R의 범위 (0,1)를 갖기때문에 편리함.
그래프 사진을 보면 0과 1 구분이 0 축에서 확실히 나뉘고 있음.
텐서플로는 시그모이드 값에 대한 교차 엔트로피 손실을 계산한는 유틸리티 함수를 제고앟ㅂ니다.
가장 간단한 함수는 tf.nn.sigmoid_corss_entropy_with_logits입니다
'''

# 단순 로지스틱 회귀 모델 정의
 #텐서플로 그래프 생성
with tf.name_scope("placeholders"):
    # 데이터포인트 x는 2차원입니다.
    x = tf.placeholder(tf.float32, (N,2))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal((2,1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
    y_logit = tf.squeeze(tf.matmul(x,W) + b)
    # 시그모이드로 클래스 1의 확률을 확인합니다.
    y_one_prob = tf.sigmoid(y_logit)
    # P(y=1) 반올림으로 정확한 예측을 합니다.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    # 각 데이터포인트에 대한 교차 엔트로피 항 계산
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    # 모든 기여의 합
    l = tf.reduce_sum(entropy)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(.01).minimize(l)
with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('train_logistic', tf.get_default_graph())

# 로지스틱 회귀 모델 학습 (선형 회귀 모델과 동일)
n_steps = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 모델 학습
    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("loss: %f" % loss)
        train_writer.add_summary(summary, i)