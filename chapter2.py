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

a = tf.zeros(2) "<tf.Tensor 'zeros:0' shape=(2,) dtype=float32>"
a.eval() "array([0., 0.], dtype=float32)"

'''텐스를 평가한 값은 파ㅣㅇ썬 객체 자체입니다. 특히 a.eval()은 numpy.ndarray 객체입니다. 
텐서플로와 넘파이 컨벤션과 상당 부분이 호환되도록 설계되었음.'''

a=tf.zeros((2,3)) 'Shape 에 튜플을 넣어주는듯.'
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
'2개의 원소배열 가지는 배열이 2개 있고, 또 그걸 2개 가지고있다.....'을

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