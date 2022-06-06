import numpy as np
import math
from typing import Union


class DecisionNode(object):
    def __init__(self, f_idx, threshold, value=None):
        self.f_idx = f_idx
        self.threshold = threshold
        self.value = value
        self.L = None
        self.R = None


class MetaLearner(object):
    def __init__(self, min_samples: int = 1, min_gain: float = 0, max_depth: Union[int, None] = None,
                 max_leaves: Union[int, None] = None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None
        self.depth = 1

    @staticmethod
    def end_tree(rst) -> DecisionNode:
        return DecisionNode(value=np.mean(rst), f_idx=None, threshold=None)

    @staticmethod
    def entropy_gain(self, y, l, r):
        l_ = y[l]
        r_ = y[r]
        a = 0
        b = 0
        if len(r_) != 0:
            a = len(r_)/len(y)*np.var(r_)
        if len(l_) != 0:
            b = len(l_)/len(y)*np.var(l_)
        return np.var(y) - a - b

    def build_tree(self, x, y, depth) -> DecisionNode:
        if depth == self.max_depth or len(x) < self.min_samples:
            return self.end_tree(y)
        best_idx = -1
        best_gain = -math.inf
        best_threshold = None
        b_ = []
        r_ = []
        for i in range(len(x[0])):
            values = np.unique(x[:, i])
            for j in range(len(values)-1):
                v = values[j]
                b = x[:, i] <= v
                r = ~b
                gain = self.entropy_gain(self, y, b, r)
                if gain >= best_gain:
                    best_gain = gain
                    best_idx = i
                    best_threshold = v
                    b_ = b
                    r_ = r
        node = DecisionNode(best_idx, best_threshold)
        if len(x[b_]) == 0:
            node.L = self.end_tree(y)
        else:
            node.L = self.build_tree(x[b_], y[b_], depth + 1)
        if len(x[r_]) == 0:
            node.R = self.end_tree(y)
        else:
            node.R = self.build_tree(x[r_], y[r_], depth+1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        selected = self.select(x)
        x_ = []
        for i in x:
            tmp = []
            for m in selected:
                tmp.append(i[m])
            x_.append(tmp)
        self.root = self.build_tree(np.array(x_), y, 1)

    def predict(self, item):
        ans = []
        for i in item:
            ans.append(self.re_predict(i))
        return ans

    def re_predict(self, item):
        node = self.root
        while node.value is None:
            if item[node.f_idx] < node.threshold:
                node = node.L
            else:
                node = node.R
        return node.value

    @staticmethod
    def select(x) -> list:
        n = len(x[0])
        k = math.floor(np.log2(n))+1
        rst = []
        while True:
            if len(rst) == k:
                return rst
            a = np.random.randint(0, n)
            if a not in rst:
                rst.append(a)
