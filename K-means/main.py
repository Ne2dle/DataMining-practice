import numpy as np
import dm2022exp
from collections import Counter
from sklearn.metrics import silhouette_score


class KMeans(object):
    def __init__(self, num_cluster: int, max_iter: int = 1000, tol: float = 1e-4):
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.depth = 1

    def fit(self, x: np.ndarray) -> np.ndarray:
        sel = np.random.choice(len(x), self.num_cluster, replace=False)
        idx = [x[i] for i in sel]
        rst = [-1 for i in range(len(x))]
        while True:
            self.re_fit(x, idx, rst)
            dic = {}
            for i in rst:
                if i not in dic.keys():
                    dic[i] = 1
                else:
                    dic[i] += 1
            new_idx = []
            for m in range(len(idx)):
                tmp = [0,0]
                for i in range(len(x)):
                    if rst[i] == m:
                        for z in range(len(x[i])):
                            tmp[z] += x[i][z]/dic[m]
                new_idx.append(tmp)
            if self.cal(idx, new_idx) < self.tol:
                return self.re_fit(x, new_idx, rst)
            else:
                self.depth += 1
                idx = new_idx
            if self.depth == self.max_iter:
                return np.array(rst)

    def re_fit(self, x, idx, rst):
        for i in range(len(x)):
            distance = 0
            flag = 0
            for m in range(len(idx)):
                if flag == 0:
                    distance = self.cal_distance(x[i], idx[m])
                    flag = 1
                    rst[i] = m
                else:
                    if self.cal_distance(x[i], idx[m]) < distance:
                        distance = self.cal_distance(x[i], idx[m])
                        rst[i] = m
        return np.array(rst)

    @staticmethod
    def cal_distance(m, n):
        s = 0
        for i in range(len(m)):
            s += (m[i] - n[i]) * (m[i] - n[i])
        return s

    @staticmethod
    def cal(m, n):
        s = 0
        for a, b in zip(m, n):
            s1 = 0
            for g, f in zip(a, b):
                s1 += np.power(g-f, 2)
            s += np.sqrt(s1)
        return s


def purity(y_pred, y, n_class):

    pr = 0
    for i in range(n_class):
        idxs = y_pred == i
        cnt = Counter(y[idxs])
        pr += cnt.most_common()[0][1]
    return pr / len(y)


if __name__ == '__main__':
    X, y = dm2022exp.load_ex4_data()
    k = KMeans(8)
    s = k.fit(X)
    # 显示准确的聚类图像
    dm2022exp.show_exp4_data(X, y)
    # 显示预测的聚类图像
    dm2022exp.show_exp4_data(X, s)
    # 打印纯度
    print(purity(s, y, 8))
    # 打印平均轮廓系数
    print(silhouette_score(X, y))
