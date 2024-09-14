import numpy as np
import matplotlib.pyplot as plt

# 퍼셉트론

"""
w = weight 가중치, theta = 임계값
y = 0 (w1*x1 + w2*x2 <= theta)
    1 (w1*x1 + w2*x2 > theta)

b = -theta = bias 편향
y = 0 (b + w1*x1 + w2*x2 <= 0)
    1 (b + w1*x1 + w2*x2 > 0)
^ 식 단순화
"""

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

"""
여기서 문제 발생!
퍼셉트론은 직선 하나로 나눈 영역만 표현 가능 => 선형 영역
XOR 게이트의 경우 곡선의 영역으로 표현 가능 => 비선형 영역

따라서 여러 퍼셉트론을 겹치는 *다층 퍼셉트론* 사용
"""

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

"""
XOR 게이트는 입력 노드 (0층), NAND와 OR 노드 (1층), 출력 노드 (2층)으로 2층 퍼셉트론
"""


# 신경망

"""
퍼셉트론 vs 신경망
퍼셉트론은 가중치 등을 수동으로 지정
신경망은 자동으로 학습을 통해 지정

신경망 (입력층 => 은닉층 => ... => 은닉층 => 출력층)
ex) 3층 신경망 (입력층 (0) => 은닉층 (1) => 은닉층 (2) => 출력층 (3))

퍼셉트론에서 편향을 x1, x2처럼 입력으로 빼고 표현하면 편향의 가중치는 1이라고 볼 수 있음
y = h(b + w1*x1 + w2*x2)
h(x) = 0 (x <= 0)
       1 (x > 0)
위 식이 신경망의 신호 전달 과정

이때 h(x)는 *활성화 함수 (activation function)*
a = b + w1*x1 + w2*x2
y = h(a)
위 식으로 표현 가능하므로 활성화 함수만 변경할 수 있음
"""


"""
계단 함수 (step function)
"""

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_np(x):
    # numpy 배열
    y = x > 0
    return y.astype(np.int64)


"""
시그모이드 함수 (sigmoid function)

h(x) = 1 / 1+e^(-x)
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
ReLU 함수 (Rectified Linear Unit function)

h(x) = x (x > 0)
       0 (x <= 0)
"""

def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function_np(x)
y2 = sigmoid(x)
y3 = relu(x)
plt.plot(x, y1, linestyle="--")
plt.plot(x, y2, linestyle="-.")
plt.plot(x, y3)
plt.ylim(-0.1, 1.1)
plt.legend(["step function", "sigmoid function", "ReLU function"])
plt.show()


"""
신경망에서는 활성화 함수가 비선형 함수여야 함
선형 함수 h(x) = cx가 있다고 했을 때 3층 신경망을 만들면
y(x) = h(h(h(x))) <=> y(x) = c*c*c*x
y(x) = ax일 때 a = c^3이므로 은닉층이 없다고 볼 수 있음

즉, 층을 쌓기 위해선 비선형 함수를 사용!
"""
