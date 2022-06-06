from sklearn.model_selection import train_test_split
from metalearner import MetaLearner as Mt
import numpy as np
from sklearn import datasets
from sklearn import ensemble
import matplotlib.pyplot as plt


class RandomForest:
    def __init__(self, n):
        self.num = n
        self.trees = [Mt(max_depth=5,min_samples=20) for i in range(n)]

    def fit(self, x, y):
        for i in range(self.num):
            x, y = self.sample(x, y)
            self.trees[i].fit(np.array(x), np.array(y))

    @staticmethod
    def sample(x, y):
        y_ = []
        x_ = []
        for i in range(len(y)):
            r = np.random.randint(0, len(y))
            y_.append(y[r])
            x_.append(x[r])
        return x_, y_

    def predict(self, x):
        rst = []
        rst_ = []
        for i in self.trees:
            tmp = i.predict(x)
            rst.append(tmp)
        rst = np.array(rst)
        for i in range(0,len(rst[0])):
            m = rst[:, i]
            rst_.append(np.mean(m))
        return np.array(rst_)


if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # 自己的算法
    # m = RandomForest(15)
    # m.fit(X_train, y_train)
    # p = m.predict(X_test)
    # print(p)
    # print("my MSE:", sum([(p[i]-y_test[i])**2 for i in range(len(p))])/len(p))

    # sklearn的算法
    skl = ensemble.RandomForestRegressor(n_estimators=15)
    skl.fit(X_train, y_train)
    p_ = skl.predict(X_test)
    print(p_)
    print("sklearn MSE is :", sum([(p_[i]-y_test[i])**2 for i in range(len(p_))])/len(p_))

    # 学习曲线
    rst = []
    for i in range(100):
        rfc = ensemble.RandomForestRegressor(n_estimators=i + 1)
        rfc.fit(X_train, y_train)
        score = rfc.score(X_test, y_test)
        rst.append(score)
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 101), rst)
    plt.show()


