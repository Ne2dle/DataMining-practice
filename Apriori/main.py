import dm2022exp
import numpy as np
import pandas as pd
import copy
import math
import collections
import functools
from sklearn.preprocessing import LabelEncoder
from typing import List, Union, Tuple


class Apriori:
    score = {}

    def __init__(self, min_sup: float = 0.2):
        self.min_sup = min_sup
        self.min_conf = 0.5

    def fit(self, X: List[List[str]]):
        f = LabelEncoder()
        k = 1
        tmp = []
        rst = []
        rule = {}
        for i in X:
            for j in i:
                if j not in tmp:
                    tmp.append(j)
        f.fit(tmp)
        for i in range(len(tmp)):
            tmp[i] = list(f.transform([tmp[i]]))
        tmp = sorted(tmp)
        new_x = list(map(set, [f.transform(i) for i in copy.deepcopy(X)]))
        tmp2 = self.scan(new_x, tmp, rst, f)

        while len(tmp2) != 0:
            tmp = self.scan2(tmp2, k)
            tmp2 = self.scan(new_x, tmp, rst, f)
            self.UpdateRule(tmp2, rule, f)
            k += 1
        return rst, rule

    def UpdateRule(self, freq, rule, f):
        for value in freq:
            element = self.get_sub_set(value)
            support = self.score[str(value)]
            for number in element:
                number_set = set(number)
                no_number_set = set(value).difference(number_set)
                number_support = self.score[str(number)]
                conf = support / number_support
                if conf >= self.min_conf:
                    temp1_set = set([i for i in f.inverse_transform([i for i in number_set])])
                    temp2_set = set([i for i in f.inverse_transform([i for i in no_number_set])])
                    rule[str(temp1_set) + '->' + str(temp2_set)] = conf

    def scan(self, dataset, ck, rst, f):
        tmp = []
        for j in ck:
            support = self.calc_sup(j, dataset)
            if support >= self.min_sup:
                rst.append([frozenset(f.inverse_transform(j)), support])
                tmp.append(j)
        return tmp

    def scan2(self, lk, k):
        new_can = []
        for i in range(len(lk)):
            for j in range(i + 1, len(lk)):
                if k == 1 or lk[i][:k - 1] == lk[j][:k - 1]:
                    new_can.append(lk[i] + lk[j][-1:])
        return new_can

    def get_sub_set(self, nums):
        sub_sets = [[]]
        for x in nums:
            sub_sets.extend([item + [x] for item in sub_sets])
            pass
        sub_sets.remove([])
        sub_sets.remove(nums)
        return sub_sets

    def calc_sup(self, x, dataset):

        if str(x) in self.score.keys():
            return self.score[x]
        count = 0
        for affair in dataset:
            if set(x).issubset(affair):
                count += 1
        self.score[str(x)] = count / len(dataset)
        return count / len(dataset)


m = Apriori(0.005)
data = dm2022exp.load_ex5_data()
rst, rule = m.fit(data)
print(rst)
print(rule)
