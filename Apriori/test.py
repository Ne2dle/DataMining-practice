import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import dm2022exp
import functools


a = [[1],[2],[3],[4],[5],[6],[1,2]]
z = [0,1,4]
count = 0
for i in z:
    i = i-count
    a.pop(i)
    count +=1

print(a)
data = dm2022exp.load_ex5_data()

m = TransactionEncoder()
m.fit(data)
m.transform(data)
df = pd.DataFrame(m.transform(data), columns=m.columns_)
ret = apriori(df, min_support=0.005, use_colnames=True, verbose=True)
print(ret)
