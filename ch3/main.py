import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from PIL import Image

warnings.filterwarnings("ignore")
mndata = MNIST("../mnist-data", gz=True, return_type="numpy")


# 신경망
# ===

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


"""
신경망에서 행렬의 곱을 활용할 수 있음 (가중치만 포함)

[x1, x2] * [
    [w11, w21, w31],
    [w12, w22, w32]
] = [x1*w11+x2*w12, x1*w21+x2*w22, x1*w31+x2*w32]
  = [y1, y2, y3]
"""

"""
이제 편향을 포함한 3층 신경망을 생각해보면
위와 비슷하지만 행렬의 곱에 편항 배열을 더하면 됨

[x1, x2] * [
    [w11, w21, w31],
    [w12, w22, w32]
] + [b1, b2, b3]
= [x1*w11+x2*w12, x1*w21+x2*w22, x1*w31+x2*w32] + [b1, b2, b3]
= [y1, y2, y3]
"""

# 입력층 (0)
X = np.array([1.0, 0.5])

# 은닉층 (1)
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(X.shape, W1.shape, B1.shape)
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

# 은닉층 (2)
W2 = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 출력층 (3)
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
identity_function = lambda x: x
Y = identity_function(A3)
print(Y)


"""
위 과정을 단순화 시키면 다음과 같음
"""

identity_function = lambda x: x

def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    # forward => 순방향으로 전달됨 (순전파)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


# 항등 함수와 소프트맥스 함수
# ===

"""
기계학습 문제는 분류 (classification)와 회귀 (regression)으로 나뉨
분류는 데이터가 어느 클래스인지 알아내는 것이고, 회귀는 입력 데이터에서 연속적인 수치를 예측하는 문제임
ex) 사물 구별 (분류), 나이 추정 (회귀)
"""

"""
위 신경망의 출력층에서는 회귀에 주로 사용되는 항등 함수 (identity function)를 사용함
항등 함수는 입력을 그대로 출력함

반면, 소프트맥스 함수 (softmax function)는 분류에 사용함
y_k = exp(a_k) / ∑(i=1, ~n) exp(a_i)
전체 입력 신호의 지수 함수의 합 분의 k번째 입력 신호의 지수 함수 (즉, 확률)
(exp는 지수 함수. 지수 함수를 사용하는 이유는 차이를 명확히 하기 위해서)
"""

"""항등 함수"""

identity_function = lambda x: x


"""소프트맥스 함수"""

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

"""
여기서 문제 발생!
지수 함수는 기하급수적으로 늘어나 입력 신호가 1000이 되기만 해도 오버플로우가 발생해 계산이 불가능함

y_k = exp(a_k) / ∑(i=1, ~n) exp(a_i)
    = C exp(a_k) / C ∑(i=1, ~n) exp(a_i)
    = exp(a_k + log C) / ∑(i=1, ~n) exp(a_i + log C)
    = exp(a_k + C') / ∑(i=1, ~n) exp(a_i + C')

위처럼 C'이 지수 함수의 입력에 더해지더라도 y_k에는 영향을 미치지 않음
따라서 C'를 통해 지수 함수의 입력을 줄여줄 필요가 있음 
"""

a = np.array([1000, 1010, 980])
y = softmax(a)
print(y)

del softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([1000, 1010, 980])
y = softmax(a)
print(y)


# 손글씨 숫자 인식
# ===

"""
MNIST 데이터셋을 이용해 손글씨 숫자 추론 과정을 구현함
이처럼 학습 과정을 생략하고 추론 과정만 구현하는 과정을 순전파 (forward propagation)라고 함

MNIST 데이터셋 => 28x28 크기의 회색조 숫자 (0~9) 이미지 집합
"""

x_train, t_train = mndata.load_training()
x_test, t_test = mndata.load_testing()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

def show_img():
    img = x_train[0]
    label = t_train[0]

    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

"""
입력층 (28*28=784) => 은닉층 (50) => 은닉층 (100) => 출력층 (0~9=10)
은닉층의 뉴런 개수는 임의로 정함
"""

def get_data():
    x_test, t_test = mndata.load_testing()
    return x_test, t_test

def init_network():
    with open("mnist_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print(f"Accuracy={float(accuracy_cnt) / len(x)}")

"""
여기선 구현되지는 않았지만, 데이터를 특정 범위로 변환하는 처리를 정규화 (normalization)라고 함
위처럼 신경망의 입력 데이터에 변환을 가하는 것을 전처리 (pre-processing)라고 함
"""


x, _ = get_data()
network = init_network()
W1, W2, W3 = network["W1"], network["W2"], network["W3"]
print(x.shape, W1.shape, W2.shape, W3.shape)

"""
X      W1        W2        W3        Y
784 => 784x50 => 50x100 => 100x10 => 10

위처럼 각 층의 배열 형상은 일치하는 묶음이 있음
위는 이미지 1장을 기준으로 한 것임

X          W1        W2        W3        Y
100x784 => 784x50 => 50x100 => 100x10 => 100x10

만약 100장 분량을 대상으로 한다면 위처럼 한 묶음이 될 것임
이처럼 하나로 묶은 입력 데이터를 배치 (batch)라고 함

작은 배열을 여러 번 계산하는 것 보다 큰 배열을 한꺼번에 계산하는 것이 더 빠름
렌더링을 할 때 타일을 사용하는 것과 비슷함
∴ 배치를 사용하면 결과 계산이 빠름
"""

x, t = get_data()
network = init_network()

batch_size = 0
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

"""
손글씨 숫자 인식을 배치를 사용해 구현함
(axis는 n번째 차원을 기준으로 최댓값을 구하는 것)
"""
