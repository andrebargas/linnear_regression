import quandl, math
import numpy as np
import pandas as pd
from pandas.DatetimeIndex import to_pydatetime
from sklearn import preprocessing,  svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
print(df['Date'])
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['variacao_dia'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['abertura'] = df['Adj. Open']
df['baixa'] = df['Adj. Low']
df['alta'] = df['Adj. High']


df_label = pd.DataFrame()
df_label['fechamento'] = df['Adj. Close' ]
df_label['volume'] = df['Adj. Volume' ]

df_test2 = df[:- 300]
df_test2_label = df_label[:- 300]
df = df[['abertura', 'baixa', 'alta', 'variacao_dia']]



# print(df_test2)
# print(df_test2_label)


# Regrecao para o fechamento
X_train, X_test, y_train, y_test = train_test_split( df, df_label['fechamento'], test_size=0.33, random_state=42)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

print("Score: ", confidence)

# Regrecao para o volume
style.use('ggplot')

result = clf.predict(df_test2)

# print(result)
# df_test2.drop(df.head(3).index , inplace=True  )
datetime = pd.to_pydatetime(df_test2.index()).dt.date
print( datetime)

# plt.plot(result[1], df_test2['Date'],  'fechamento_previsto', color="blue")

plt.title("Mais incrementado")

plt.grid(True)
plt.xlabel("eixo horizontal")
plt.ylabel("que legal")
plt.show()
