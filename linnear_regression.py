import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing,  svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


def generateLRfechamento():
    df = quandl.get("WIKI/GOOGL")
    arr_dates = df.index.values, type(df)
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    df['variacao_dia'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['abertura'] = df['Adj. Open']
    df['baixa'] = df['Adj. Low']
    df['alta'] = df['Adj. High']

    df_label = pd.DataFrame()
    df_label['fechamento'] = df['Adj. Close' ]
    df_label['volume'] = df['Adj. Volume' ]

    df_test2 = df[:- 300]
    df_test2 = df[['abertura', 'baixa', 'alta', 'variacao_dia']]
    df_test2_label = df_label[:- 300]
    df = df[['abertura', 'baixa', 'alta', 'variacao_dia']]

    # Linnear Regression para fechamento
    X_train, X_test, y_train, y_test = train_test_split( df, df_label['fechamento'], test_size=0.33, random_state=42)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("Score: ", confidence)
    # # Plot para Regressao Aberturax
    #
    # style.use('ggplot')
    # plt.plot(df_test2['abertura'] , result, color='blue' )
    # plt.grid(True)
    # plt.xlabel("Abertura")
    # plt.ylabel("Fechamento")
    # plt.savefig("fechamento_regression_image.png")
    #Plot para previsão do teste
    result = clf.predict(df)
    print(df.index.values.astype(np.int64))
    style.use('ggplot')
    line1 = plt.plot(df.index.values.astype(np.int64) , df_label['fechamento'], color='red' )
    line2 = plt.plot(df.index.values.astype(np.int64) , result, color='blue' )
    plt.grid(True)
    plt.xlabel("Ano em TimeStamp")
    plt.ylabel("Fechamento")
    # plt.axis([1292960000000000000,1592960000000000000,400,1000])
    plt.show()

# Linnear Regression para volume
def generateLRvolume():
    df = quandl.get("WIKI/GOOGL")
    arr_dates = df.index.values, type(df)
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    df['variacao_dia'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['abertura'] = df['Adj. Open']
    df['baixa'] = df['Adj. Low']
    df['alta'] = df['Adj. High']

    df_label = pd.DataFrame()
    df_label['fechamento'] = df['Adj. Close' ]
    df_label['volume'] = df['Adj. Volume' ]

    df_test2 = df[:- 300]
    df_test2 = df[['abertura', 'baixa', 'alta', 'variacao_dia']]
    df_test2_label = df_label[:- 300]
    df = df[['abertura', 'baixa', 'alta', 'variacao_dia']]

    X_train, X_test, y_train, y_test = train_test_split( df, df_label['volume'], test_size=0.33, random_state=42)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print("Score: ", confidence)

    # # Plot para Regressao Aberturax
    #
    # style.use('ggplot')
    # plt.plot(df_test2['abertura'] , result, color='blue' )
    # plt.grid(True)
    # plt.xlabel("Abertura")
    # plt.ylabel("Fechamento")
    # plt.savefig("fechamento_regression_image.png")

    #Plot para previsão do teste

    result = clf.predict(df)

    print(df.index.values.astype(np.int64))

    style.use('ggplot')
    plt.plot(df.index.values.astype(np.int64) , df_label['volume'], color='red' )
    plt.plot(df.index.values.astype(np.int64) , result, color='blue' )
    plt.grid(True)
    plt.xlabel("Ano em TimeStamp")
    plt.ylabel("Fechamento")

    # plt.axis([1292960000000000000,1592960000000000000,400,1000])
    plt.show()

if __name__ == '__main__':
    generateLRfechamento()
    generateLRvolume()
