import numpy as np


# 퍼셉트론
# ===

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
