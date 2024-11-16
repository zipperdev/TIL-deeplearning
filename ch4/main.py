import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


"""
딥러닝 = 종단간 기계학습 (e2e machine learning)
종단간 기계학습은 인간이 처음부터 끝까지 개입하지 않음을 의미

기계학습은 훈련 데이터와 시험 데이터로 나눠 학습과 실험 수행
범용 능력은 접하지 못한 데이터를 풀어내는 능력
범용 능력을 제대로 평가하기 위해 데이터를 훈련/시험 데이터로 쪼갬

오버피팅은 특정 데이터셋에만 지나치게 최적화된 상태를 의미
기계학습 시 오버피팅을 피해야함
"""


# 손실 함수
# ===

"""
손실 함수 (loss function)는 신경망 학습에서 사용하는 신경망 성능의 나쁨을 나타내는 지표
"""


"""
오차제곱합 (Sum of Squares for Error, SSE)

E = 1/2 ∑(k) (y_k - t_k)^2
k는 데이터의 차원 수, y_k는 신경망의 출력, t_k는 정답 레이블
"""

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


"""
교차 엔트로피 오차 (Cross Entropy Error, CEE)

E = -∑(k) t_k log y_k
"""

def cross_entropy_error(y, t):
    delta = 1e-7  # np.log(0) = -inf 방지
    return -np.sum(t * np.log(y + delta))


# 2가 정답
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# 2일 확률이 높다고 추정
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(sum_squares_error(y, t))
print(cross_entropy_error(y, t))

# 7일 확률이 높다고 추정
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(sum_squares_error(y, t))
print(cross_entropy_error(y, t))


# 미니배치와 손실 함수

"""
배치는 여러 데이터를 한 번에 처리하므로 단일 배치에 대해 손실 함수를 적용해야됨

E = -1/N ∑(n) ∑(k) t_nk log y_nk
위는 데이터 N개 배치에 대한 평균 교차 엔트로피 오차

이때 전체 데이터에 대해 계산하는 것은 비현실적이므로, 일부만 골라 학습함
이 일부를 미니배치 (mini-batch), 이러한 학습 방법을 미니배치 학습이라고 함
"""

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 100
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(* np.log(y[np.arange(batch_size), t] + 1e-7))


"""
왜 정확도가 아닌 손실 함수의 값을 사용할까?

신경망 학습에서는 미분을 사용함
정확도는 매개변수에 약간의 변화를 주면 불연속적으로 미세하게 변함
따라서 정확도를 지표로 사용하면 미분의 값이 대부분 0에 가깝기 때문에 발전할 수 없음

비슷한 이유로 활성화 함수로 비선형 함수를 사용함
"""


# 미분
# ===

# 수치 미분

"""
해석적 미분을 그대로 구현한다면 다음과 같음
하지만 h가 반올림 오차로 표현이 부정확할 수 있고, y의 차에 오차가 생기기도 함 
"""

def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h

"""
그러므로 컴퓨터에서 미분은 개선이 필요함

h를 10^-4 정도로 사용하는 것이 적당함
이제 h의 값이 0에 덜 근접하므로 함수 f의 차분에 오차가 발생함
따라서 (x + h)와 (x - h)일 때의 함수 f의 차분, 중심 차분을 사용함
((x + h)와 x일 때의 차분을 전방 차분이라 함)

위 개선을 적용한 미분을 수치 미분이라 함  
"""

del numerical_diff
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


function_1 = lambda x: 0.01 * x ** 2 + 0.1 * x
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y)
plt.plot(x, y2, linestyle="--")
plt.show()


# 편미분

"""
편미분 = 변수가 여럿인 함수에 대한 미분

f(x_0, x_1) = x_0^2 + x_1^2, x_0 = 3, x_1 = 4일 때, x_0에 대한 편미분 δf/δx_0를 구해보자.
"""

function_2 = lambda x: x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0**2.0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3))

"""
위와 같이 수치 미분과 비슷하지만, 변수를 고정해서 풀 수 있음
"""


# 기울기

"""
위에선 변수가 하나로 고정된 편미분을 품
x_0, x_1의 편미분을 동시에 계산하는 경우도 있음
이때 (δf/δx_0, δf/δx_1)처럼 모든 변수의 편미분을 백터로 정리한 것을 기울기 (gradient)라고 함
"""

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        # f(x + h)의 값
        x[idx] = tmp + h
        fxh1 = f(x)

        # f(x - h)의 값
        x[idx] = tmp - h
        fxh2 = f(x)

        # 수치 미분
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp

    return grad

# f(x_0, x_1) = x_0^2 + x_1^2, (x_0, x_1) = (3, 4)일 때의 기울기
print(numerical_gradient(function_2, np.array([3.0, 4.0])))


"""
기울기 => 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표
그러나 대부분의 복잡한 함수에서는 기울기가 가리키는 곳이 최솟값이란 보장은 없음
"""


# 경사법
# ===


"""
경사법 (gradient method) = 현 위치에서 기울어진 방향으로 일정 거리만큼 이동을 반복하며 기울기를 구하해 함수의 값을 점차 줄이는 것

최솟값을 찾는다면 경사 하강법 (gradient descent method), 최댓값을 찾는다면 경사 상승법 (gradient ascent method)이라고 함
신경망 학습에서 경사법을 많이 사용함

x_0 = x_0 - 𝝶 δf/δx_0
x_1 = x_1 - 𝝶 δf/δx_1

𝝶 (에타) = 갱신하는 양
신경망에서는 𝝶를 학습률 (learning rate)라고 함 
학습률과 같은 매개변수는 학습할 수 없어 직접 지정하므로 하이퍼파라미터 (hyper parameter)라고 부름
"""

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def gradient_descent_history(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

x, x_history = gradient_descent_history(function_2, np.array([-3.0, 4.0]), lr=0.1, step_num=40)

plt.plot([-5, 5], [0, 0], "--b")
plt.plot([0, 0], [-5, 5], "--b")
plt.plot(x_history[:, 0], x_history[:, 1], "o")

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()
