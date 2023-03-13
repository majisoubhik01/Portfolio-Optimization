!pip install pandas_datareader scikit-learn matplotlib cvxopt seaborn yahoofinancials
import streamlit as st
import time
import pandas as pd
import pandas_datareader as dr
import numpy as np
from yahoofinancials import YahooFinancials
from datetime import date
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import date

#Import Model Packages 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

#Package for optimization of mean variance optimization
import cvxopt as opt
from cvxopt import blas, solvers

#progress_text = "Operation in progress. Please wait."
#my_bar = st.progress(0, text=progress_text)

#for percent_complete in range(100):
   #time.sleep(0.02)
   #my_bar.progress(percent_complete + 1, text=progress_text)
#st.title("Hello Streamlit")
st.image('Logo_header@2x.jpg')
#sector = st.selectbox('Pick sector', ['IT', 'AUTO', 'OIL&GAS'])
algo = st.selectbox('Pick algo', ['HRP', 'MVP'])
st.write(algo)

amt = st.text_input('Enter amount:',10000)
st.write(amt)
#dataset = pd.DataFrame({
 #   a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})

dataset = pd.read_csv("autoret.csv")
#missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

#missing_fractions.head(10)

#drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

#dataset.drop(labels=drop_list, axis=1, inplace=True)

#dataset=dataset.fillna(method='ffill')

X = dataset.copy()
row= len(X)
train_len = int(row*.8)
#train_len = int(row*.8846)

X_train = dataset.head(train_len)

X_test = dataset.tail(row-train_len)

returns =  pd.read_csv("autoret.csv") #X_train.pct_change()
returns_test =  pd.read_csv("autoret_test.csv") #X_test.pct_change()

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    return dist

#Calulate linkage
dist = correlDist(returns.corr())
link = linkage(dist, 'ward')
#link[0]

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar



def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w

def getMVP(cov):

    cov = cov.T.values
    n = len(cov)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(cov)
    #pbar = opt.matrix(np.mean(returns, axis=1))
    pbar = opt.matrix(np.ones(cov.shape[0]))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    
    # Calculate efficient frontier weights using quadratic programming
    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER    
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO    
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return list(wt)

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    #plt.figure(figsize=(20, 10))
    #dn = sch.dendrogram(link, labels=cov.index.values)
    #plt.show()
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()

def get_req_portfolios(returns):
    cov, corr = returns.cov(), returns.corr()
    if algo == 'HRP':
        hrp = getHRP(cov, corr)
        portfolios = pd.DataFrame([hrp], index=['HRP']).T
    else:
        mvp = getMVP(cov)
        mvp = pd.Series(mvp, index=cov.index)
        portfolios = pd.DataFrame([mvp], index=['CLA']).T
    #portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T
    return portfolios

portfolios = get_req_portfolios(returns)

portfolios.iloc[:,0] = round(portfolios.iloc[:,0]*int(amt),2)
st.table(portfolios.iloc[:,0])
