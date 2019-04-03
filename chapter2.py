import tensorflow as tf

''' 대화형으로 텐서플로를 실험할 때는 tf.InteractiveSession() 을 사용하면 편리합니다
 이 구분을 실행하면 대부분 명령형으로 실행할 수 있어서 초보자는 텐서플로를 더욱 쉽게 사용할 수 있습니다.
 명령형과 선언형 방식의 차이는 이 장 뒤에 자세히..'''
tf.InteractiveSession()

'''텐서플로는 기본 텐서를 메모리에 인스턴스화 할 수 있는 여러 함수를 제공합니다. 가장 가단하게 tf.zeros() 와 tf.ones() 가 있습니다.
tf.zeros() 는 텐서 형상(파이선 튜플로 표현)을 받아 0으로 채워진 해당 형상의 텐서를 반화합니다..'''
tf.zeros(2)
'''텐서 그 자체의 값이 아닌 텐서의 참조를 반환합니다. 강제로 텐서 값을 ㅏㄴ환하려면 텐서 객체의 tf.Tensor.eval() 메서드를 사용하면 됩니다.
tf.InteractiveSession() 으로 초기화한 상태이므로 이 메서드는 0으로 이루어진 텐서 값을 반환합니다.'''

a = tf.zeros(2) #"<tf.Tensor 'zeros:0' shape=(2,) dtype=float32>"
a.eval() #"array([0., 0.], dtype=float32)"

'''텐스를 평가한 값은 파ㅣㅇ썬 객체 자체입니다. 특히 a.eval()은 numpy.ndarray 객체입니다. 
텐서플로와 넘파이 컨벤션과 상당 부분이 호환되도록 설계되었음.'''

a=tf.zeros((2,3)) #'Shape 에 튜플을 넣어주는듯.'
a.eval()
'''array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)''' '2행, 3열의 2차원 배열이 모두 0으로 채워져 있는 상태로 나왔다. 타입은 float32 이래서 마지막에 숫자. 이 붙는다.'
'3개의 원소를 가지는 배열이 2행으로(2개)'

b=tf.ones((2,2,2))
b.eval()
'''array([[[1., 1.],
        [1., 1.]],
       [[1., 1.],
        [1., 1.]]], dtype=float32)'''
'3차원부터는 읽기가 힘들다...'
'잘 보자.. 1행 1열의 원소는 2크기의 배열 [1,1] 이 들어있다고 생각하면 이해가 되는듯.'
'2개의 원소배열 가지는 배열이 2개 있고, 또 그걸 2개 가지고있다.....'

'''0,1 외에 다른 숫자들은 tf.fill() 메소드로 채운다.'''
b=tf.fill((2,2), value=5.)
b.eval()
"""array([[5., 5.],
       [5., 5.]], dtype=float32)
"""

'''tf.constant는 tf.fill 과 유사하지만 프로그램 실행 중에 변경할 수 없는 텐서를 생성할 수 있습니다.'''
a=tf.constant(3)
a.eval()
"""3"""

'''아이디어를 실험할 때 보통 난수로 텐서를 초기화. tf.random_normal로 평균과 표준편차를 지정한 정규분포(normal distribution)로부터
추출한 값을 특정 형상의 텐서의 원소로 채울 수 있습니다.'''

a = tf.random_normal((2,2), mean=0, stddev=1)
a.eval()
"""array([[-0.16488627,  0.8210673 ],
       [ 1.0637171 , -1.6114944 ]], dtype=float32)"""

'''정규분포에서 수천만 난수를 추출할 때 몇개는 평균과 큰 차이가 있을수 있습니다. 이러한 큰 샘플이 결국 수치적 불안정성(numerical instability)로 이어질 수 있으므로
tf.random_normal() 보다는 tf.truncated_normal()을 흔히 사용합니다(절단정규분포).
이 함수는 API 측면에서 tf.random_normal() 과 동일하지만 평균에서 표준편차의 2배 이상 차이 나는 값은 모두 제외하고 다시 추출합니다.

tf.random_uniform()도 tf.random_normal()과 유사하지만 지정한 범위에 걸친 균등분포에서 난수를 추출한다는 점이 다릅니다.'''

a = tf.random_uniform((2,2), minval=-2, maxval=2)
a.eval()


"2.2.4 텐서 추가와 확장 "

''' 기본적으로 파이썬 연산자를 오버로딩해 텐서 연산에 사용하도록 만듬'''

c = tf.ones((2,2))
d = tf.ones((2,2))
e = c+d
e.eval()
"""array([[2., 2.],
       [2., 2.]], dtype=float32)"""
f=2*e
f.eval()
"""array([[4., 4.],
       [4., 4.]], dtype=float32)"""

'''텐서곱도 되지만 텐서를 곱하는 것인 행렬곱이 아니라 원소 단위 곱임을 유의해야한다.'''

c = tf.fill((2,2),2.)
d = tf.fill((2,2),7.)
e = c*d
e.eval()
"""array([[14., 14.],
       [14., 14.]], dtype=float32)"""


"2.2.5 행렬 연산"
'''텐서플로는 행렬 다루는게 많음. 단위행렬. tf.eye()로 만듬'''

a = tf.eye(4)
a.eval()
"""array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]], dtype=float32)"""

'''대각행렬. tf.range(start, limit, delta)와 tf.diag(diagonal)로 만듬'''

r = tf.range(1,5,1)
r.eval()
"""array([1, 2, 3, 4])"""
d=tf.diag(r)
d.eval()
"""array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])"""

''' 행렬전치는 tf.matrix_transpose()'''

a = tf.ones((2,3))
a.eval()
"""array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)"""
at = tf.matrix_transpose(a)
at.eval()
"""array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)"""

'''한 쌍의 행렬곱 수행. tf.matmul()'''

a = tf.ones((2,3))
a.eval()
"""array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)"""
b = tf.ones((3,4))
b.eval()
"""array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)"""
c =  tf.matmul(a,b)
c.eval()
"""array([[3., 3., 3., 3.],
       [3., 3., 3., 3.]], dtype=float32)"""


"2.2.6 텐서 형"

'''지금까지 dtype. 텐서플로에서 텐서는 tf.float32, tf.float64, tf.int32, tf.int64 등의 다양한 type이 있음.
텐서 생성 함수에 dtype을 설정하면 텐서를 특정 형으로 생성 가능.
기존 텐서에는 tf.to_double(), tf.to_float(), tf.to_int32(), tf.to_int64() 등의 형 변환 함수를 사용할 수 있습니다.'''

a=tf.ones((2,2),dtype=tf.int32)
a.eval()
"""array([[1, 1],
       [1, 1]])"""
b = tf.to_float(a)
b.eval()
"""array([[1., 1.],
       [1., 1.]], dtype=float32)"""


"2.2.7 텐서 형상 조작"

'''텐서는 단순한 숫자 집합. 다른 형상의 텐서로 바꿔 볼 수도 있다. tf.reshape()로 다양한 형상으로 변환가능'''

a=tf.ones(8)
a.eval()
"""array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)"""
b = tf.reshape(a,(4,2))
b.eval()
"""array([[1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)"""
c=tf.reshape(a,(2,2,2))
c.eval()
"""array([[[1., 1.],
        [1., 1.]],
       [[1., 1.],
        [1., 1.]]], dtype=float32)"""


'''모든 조작은 tf.reshape()로 가능. 단순한 형상 조작은 tf.expand_dims 나 tf.squeeze 함수를 이용.
tf.expand_dims는 크기 1의 차원을 텐서에 추가하여 텐서의 랭크를 1 증가시킵니다(예를 들어 랭크-1 벡터를 랭크-2인 행 벡터나 열 벡터로 변환할 때 유용합니다.)
tf.squeeze는 텐서의 크기 1인 모든 차원을 제거합니다. 행 또는 열 벡터를 납작한 벡터(flat vector)로 변환하는데 유용합니다.
tf.Tensor.get_shape() 텐서의 형상을 조회'''

a = tf.ones(2)
a.get_shape()
"""TensorShape([Dimension(2)])"""
a.eval()
"""array([1., 1.], dtype=float32)"""
b=tf.expand_dims(a,0)
b.get_shape()
"""TensorShape([Dimension(1), Dimension(2)])"""
b.eval()
"""array([[1., 1.]], dtype=float32)"""
c=tf.expand_dims(a,1)
c.get_shape()
"""TensorShape([Dimension(2), Dimension(1)])"""
c.eval()
"""array([[1.],
       [1.]], dtype=float32)"""
d = tf.squeeze(b)
d.get_shape()
"""TensorShape([Dimension(2)])"""
d.eval()
"""array([1., 1.], dtype=float32)"""


"2.2.8 브로드캐스팅"

'''브로드캐스팅(broadcasting)은 텐서 시스템에서 행렬과 다른 크기의 벡터를 서로 더할 때 쓰이는 용어입니다(넘파이에서 도입된 개념입니다.).
이 규칙을 따르면 행렬의 모든 행에 벡터르 더할 때 편리합니다. 브로드캐스팅 규칙은 상당히 복잡해서 이 규칙에 대해 정식으로 논의하지는 않을 것입니다.
브로드 캐스팅 동작법을 실험하고 살펴보는것이 보통 더 쉽습니다.'''

a=tf.ones((2,2))
a.eval()
"""array([[1., 1.],
       [1., 1.]], dtype=float32)"""
b=tf.range(0,2,1,dtype=tf.float32)
b.eval()
"""array([0., 1.], dtype=float32)"""
c=a+b
c.eval()
"""array([[1., 2.],
       [1., 2.]], dtype=float32)"""

'''벡터 b를 행렬 a의 모든 행에 더했습니다. 주목할 점은 명시적으로 b에 dtype을 지정했다는 점입니다.
dtype을 지정하지 않으면 텐서플로는 타입 에러를 보고합니다.
밑은 dtype을 지정하지 않았을때..'''

b=tf.range(0,2,1)
b.eval()
"""array([0, 1])"""
'c=a+b'
"""ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: 'Tensor("range_2:0", shape=(2,), dtype=int32)'"""

'''C 언어와 달리 텐서플로는 내부적으로 암시적 형 변환을 하지 않습니다. 산술 계산을 할 때는 보통 명시적 형 변환이 필요합니다.'''


"2.3 명령형과 선언형 프로그래밍"

'''컴퓨터 과학에서는 대부분 명령형 프로그래밍을 사용합니다. (지금까지 위에도 다 명령형으로 한거임) 간단한 프로그램을 생각하자'''

a=3
b=4
c=a+b
c
"""7"""

'''프로그램이 컴퓨터에게 명시적으로 어떠한 작업을 하라고 시키기 때문에 이런 유형의 프로그래밍을 명령형(imperative) 이라고 부릅니다.
또 다른 유형의 프로그래밍은 선언형(declarative)입니다. 선언형 시스템에서 컴퓨터 프로그램은 실행할 계산을 고수준으로만 기술합니다.
즉 컴퓨터에게 정확하게 계산하는 방법을 지시하지 않습니다. 아래 예제는 위 예제의 텐서플로 버전'''

a = tf.constant(3)
b = tf.constant(4)
c = a+b
c
"""<tf.Tensor 'add_4:0' shape=() dtype=int32>"""
c.eval()
"""7"""

'''c가 7이 아님에 주목해라. c는 단지 텐서를 나타낼 뿐. 실제 계산은 c.eval() 함수를 호출하기 전까지는 수행하지 않음.'''

"2.3.1 텐서플로 그래프"

'''텐서플로 계산은 tf.Graph 객체의 인스턴스로 표현됨. 이 그래프는 tf.Tensor 객체와 tf.Operation 객체 인스턴스로 구성됨.
tf.Tensor는 앞에서 다뤘던거고, tf.Operation은 tf.matmul같은 연산을 호출하면 tf.Operation 인스턴스가 생성되어 행렬곱 연산 수행이 필요하다고 내부적으로 표시됩니다.'''

'''tf.Graph를 명시적으로 지정하지 않으면 텐서플로는 감춰진 전역 tf.Graph 인스턴스에 텐서와 연산을 추가합니다. 이 인스턴스는 tf.get_default_graph()를 사용해 가져올 수 있습니다.'''

tf.get_default_graph()
"""<tensorflow.python.framework.ops.Graph object at 0x000001BAD29F4978>"""


"2.3.2 텐서플로 세션"

'''텐서플로에서 tf.Session() 객체는 수행할 계산 문맥을 저장. 이번 장의 처음에 tf.InteractiveSession() 를 새용해서 모든 텐서플로 계산을 위한 환경을 설정했습니다.
이를 호출해서 수행할 모든 계산에 필요한 숨겨진 전역 문맥을 생성했습니다.
그리고 tf.Tensor.eval()을 사용해 선언형으로 지정한 계산을 실행했습니다.
이를 호출하면 내부적으로 숨겨진 전역 tf.Session 문맥 안에서 평가가 이뤄집니다.
물론 숨겨진 문맥 대신 명시적 문맥을 사용해 계산하는 것이 편리할 때도 있습니다.'''

'#명시적으로 텐서플로 세션 조작'
sess = tf.Session()
a = tf.ones((2,2))
b = tf.matmul(a,a)
b.eval(session=sess)
"""array([[2., 2.],
       [2., 2.]], dtype=float32)"""

'''이 코드는 숨겨진 전역 세션 대신 sess 문맥에서 b를 평가합니다. 사실 다른 식으로 이것을 좀 더 명시화할 수 있습니다.'''

'#세션에서 계산 실행'
sess.run(b)
"""array([[2., 2.],
       [2., 2.]], dtype=float32)"""

'''사실 b.eval(session=sess) 호출은 sess.run(b)호출의 문법적 설탕(syntactic sugar)'''

'''이러한 전반적인 논의가 궤변처럼 보일지도 모릅니다. 다른 메소드들이 동일한 값을 반한하는데 어떤 세션이 동작하는지가 뭐가 중요????
명시적 세션은 실제 계산을 수행하기 전까지 값을 보여주지 않는데, 다음 절을 보자..'''


"2.3.3 텐서플로 변수"

''' 이번 절의 보든 예제 코드는 상수 텐서를 사용했다. 지금까지는 새로 만들기만 했지, 값 변경은 없었다.
지금까지 유형은 함수형(functional)이였지만 상태적(stateful)이진 않았다. 함수형 계산은 유용하지만 머신러닝은 종종 계산 상태에 매우 의존적.
학습 알고리즘은 제공된 데이터를 설명하기 위해 저장된 데이터를 갱신하는것이 핵심 규칙. 갱신 못하면 학습이 매우매우매우 어려울거야'''

'''tf.Variable() 클래스는 상태 계산이 가능하도록 텐서를 감싸는 래퍼를 제공합니다. 이 변수 객체는 텐서의 보유자(hloder) 역할을 합니다.'''

'# 텐서플로 변수 생성'
a=tf.Variable(tf.ones((2,2)))
a
"""<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32_ref>"""
# 앞에서 했던
# a=tf.ones((2,2))
# a
# <tf.Tensor 'ones_14:0' shape=(2, 2) dtype=float32>''
# 의 결과와 조금 다르다.

'''이걸 변수 a를 텐서인 것처럼 평가하려고 하면???'''

# 초기화되지 않은 변수 평가 실패
"""a.eval()"""
"""tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable_1
	 [[Node: _retval_Variable_1_0_0 = _Retval[T=DT_FLOAT, index=0, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable_1)]]"""

'''변수를 명시적으로 초기화하지 않았으므로 평가는 실패합니다. tf.global_variables_initializer를 호출하면 모든 변수를 가장 쉽게 초기화할 수 있습니다.
세션에서 이 션산을 실행하면 프로그램의 모든 변수를 초기화할 수 있습니다.'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a.eval(session=sess) # 같은 문법은 sess.run(a) 겠지?
"""array([[1., 1.],
       [1., 1.]], dtype=float32)"""

'''초기화한 이후에는 일반 텐서처럼 변수 안에 저장된 값을 가져올 수 있습니다. a=tf.Variable(tf.ones((2,2))) 로 선언했음을 기억하자.
지금까지는 일반 텐서와 변수 사이에 큰 차이는 없었지만, 변수를 할당하면 비로소 재밌어짐.
tf.assign()으로 할당을 할 수 있습니다. tf.assign()을 사용해서 존재하는 변수의 값을 갱신해보자.'''

sess.run(a.assign(tf.zeros((2,2))))
"""array([[0., 0.],
       [0., 0.]], dtype=float32)"""
sess.run(a)
"""array([[0., 0.],
       [0., 0.]], dtype=float32)"""

'''a 변수에 (2,2) 형상이 아닌 것을 실행하려고 하면?'''

'''sess.run(a.assign(tf.zeros((3,3))))'''
"""ValueError: Dimension 0 in both shapes must be equal, but are 2 and 3. Shapes are [2,2] and [3,3]. for 'Assign_1' (op: 'Assign') with input shapes: [2,2], [3,3]."""

'''변수의 형상은 초기화할 때 고정되며 갱신할때도 보존되어야 한다.
다른 흥미로운 점은 tf.assign은 내장 tf.Graph 전역 인스턴스의 일부라는 점입니다. 아까 계산하는 거라고 했었지?
텐서플로 프로그램은 매 실행마다 내부 상태를 갱신할 수 있습니다.
앞으로 이런 특성을 자주 사용할거다...'''


"2.4 마치며"

'''간단하게 봤다.
다음장에서는 텐서플로를 사용해 간단한 선형 및 로지스틱 회귀 학습 모델을 구축하는 방법을 배웁니다. 그 이후 장에서는 이러한 기반 위에서 더욱 정교한 모델을 훈련하는 방법을 알게 됩니다.'''