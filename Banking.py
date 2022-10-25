#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd 
import itertools 
import numpy as np

from pypfopt.expected_returns import mean_historical_return
from pypfopt import EfficientFrontier

import warnings
warnings.filterwarnings("ignore")


# ## Read the data from csv file

# In[2]:


data = pd.read_csv("Data.csv")


# ### Explore the data and rename columns

# In[3]:


data.head()


# - BMP, DHG, FPT, GAS, HAG, HCM, IMP, MSN, VCB, VIC

# ## Rename the columns

# In[4]:


columns = ["Ticker", "YYYYMMDD", "Open", "High", "Low", "Close", "Volume"]
data.columns = columns


# ## Read the data again for check

# In[5]:


data.head()


# In[6]:


data["Ticker"].describe()


# ## Take out the ten ticker that we need

# In[7]:


top_ten = ["BMP", "DHG", "FPT", "GAS", "HAG", "HCM", "IMP", "MSN", "VCB", "VIC"]
data_10 = data[data["Ticker"].isin(top_ten)]
data_10.reset_index(inplace = True)
data_10 = data_10.drop("index", axis = 1)

data_10.head()


# #### Check the number of tickers again

# In[8]:


data_10["Ticker"].unique() # doc nhat 


# ## Take out the tickers from the year 2016 to 2022

# In[9]:


data_10['YYYYMMDD'] = data_10['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
year = [i for i in range(2016, 2023)]

result = data_10[(data_10["YYYYMMDD"].dt.year.isin(year))] 

result.head()


# ## Just copying the data to prevent us from losing data during proccessing 

# In[10]:


pre = result.copy()

pre["day"] = pre["YYYYMMDD"].dt.day
pre["month"] = pre["YYYYMMDD"].dt.month
pre["year"] = pre["YYYYMMDD"].dt.year

pre = pre.drop(columns = ["Close"])


# ## Take out the First Notice Day of each month (from 2016 to 2022) for tickers

# In[11]:


first_day = pre.groupby(by =  ["Ticker", "year","month"]).agg('min')

first_day = first_day.reset_index(col_level = [1, 2])
first_day = first_day.drop(["year", "month", "day", "Open", "High", "Low", "Volume"], axis = 1)

indexMonth = first_day[(first_day['YYYYMMDD'].dt.year == 2016) & (first_day['YYYYMMDD'].dt.month <= 8) ].index
indexMonth2 = first_day[(first_day['YYYYMMDD'].dt.year == 2022) & (first_day['YYYYMMDD'].dt.month == 9) ].index

first_day = first_day.drop(indexMonth)
first_day = first_day.drop(indexMonth2)


# In[12]:


first_day = pd.merge(first_day, result, on = ["YYYYMMDD", "Ticker"])

first_day.head()


# ## Just take the close value

# In[13]:


first_day = first_day.drop(labels = ["Open", "High", "Low", "Volume"], axis = 1)
test = first_day.copy()


# In[14]:


first_day["Return"] = first_day.groupby("Ticker").Close.pct_change()
first_day.head()


# ## Mean return of each ticker

# In[15]:


mean_data = first_day.groupby("Ticker").agg({"Return": "mean"})
mean_data.columns = ["Mean"]
mean_data.T


# ## Var data

# In[16]:


var_data =  first_day.groupby("Ticker")["Return"].var(ddof = 0)
var_data = var_data.to_frame()
var_data.columns = ["Var"]

var_data.T


# ## Standard Deviation 

# In[17]:


std_data = first_day.groupby("Ticker")["Return"].std(ddof = 0)
std_data = std_data.to_frame()
std_data.columns = ["Std"]

std_data.T


# ## Correlation of Return

# In[18]:


corr_table = first_day.set_index(['Ticker', 'YYYYMMDD']).Return.unstack(['Ticker']).corr()

corr_table


# ## Covariance of Return

# In[19]:


first_day.fillna(0, inplace = True)

cov_table = first_day.set_index(['YYYYMMDD', "Ticker"]).Return.unstack("Ticker").cov(ddof = 1)

cov_table


# ### E(r) and 𝞼 data

# In[20]:


raito = pd.concat([mean_data.T, std_data.T], ignore_index = False)

raito = raito.rename_axis(None, axis=1)

rf = [20/1200, 0]
raito.insert(loc = 0, column='rf', value = rf)

raito.index = ["E(r)", "𝞼"]

raito


# ### A close view to data

# In[21]:


ticker = list(result["Ticker"].unique())
combination = list(itertools.combinations(ticker, 3))


# In[22]:


df = pd.DataFrame({"Combination":combination})
df["Combination"] = df["Combination"].apply(lambda x: '-'.join(x))


# In[23]:


df["wi"] = 1/3
df["wj"] = 1/3
df["wk"] = 1/3

df["Total"] = 1


# In[24]:


df["Stock i"] = df["Combination"].apply(lambda x: x.split('-')[0])
df["Stock j"] = df["Combination"].apply(lambda x: x.split('-')[1])
df["Stock k"] = df["Combination"].apply(lambda x: x.split('-')[2])


# In[25]:


df["E(ri)"] = raito.loc["E(r)"][df["Stock i"]].values
df["E(rj)"] = raito.loc["E(r)"][df["Stock j"]].values
df["E(rk)"] = raito.loc["E(r)"][df["Stock k"]].values


# In[26]:


df["𝞼(ri)"] = raito.loc["𝞼"][df["Stock i"]].values
df["𝞼(rj)"] = raito.loc["𝞼"][df["Stock j"]].values
df["𝞼(rk)"] = raito.loc["𝞼"][df["Stock k"]].values


# In[27]:


df["Cov(i,j)"] = [cov_table[i][j] for i, j in zip(df["Stock i"], df["Stock j"])]
df["Cov(i,k)"] = [cov_table[i][k] for i, k in zip(df["Stock i"], df["Stock k"])]
df["Cov(j,k)"] = [cov_table[j][k] for j, k in zip(df["Stock j"], df["Stock k"])]


# In[28]:


df.head()


# ## Optimization

# ### Prepare the data for optimization

# In[29]:


df_new = test.set_index(['YYYYMMDD', "Ticker"]).Close.unstack(['Ticker'])

df_new = df_new.pct_change()
df_new = df_new.dropna()

df_new.head()


# ### New weight for sharpe ratio

# In[30]:


res = []

for i in range(0, 120):
    df_assets =  df_new.loc[:,[df["Stock i"][i], df["Stock j"][i], df["Stock k"][i]]]
    df_cov = df_new.loc[:,[df["Stock i"][i], df["Stock j"][i], df["Stock k"][i]]].cov()
    
    retornos1 = mean_historical_return(df_assets, returns_data = True, frequency = 2.5)
    
    ef = EfficientFrontier(retornos1, df_cov, weight_bounds = (0.05, 1))
    weights = ef.max_sharpe(risk_free_rate = 20/1200) 
    cleaned_weights = ef.clean_weights() 
    
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']
    
    res.append(weights_df["weights"].to_list())
    
    df["wi"][i] = res[i][0]
    df["wj"][i] = res[i][1]
    df["wk"][i] = res[i][2]


# ### Apply new weight for sharpe ratio and re-calculate

# In[31]:


df["E(rp)"] = df["wi"] * df["E(ri)"] + df["wj"] * df["E(rj)"] + df["wk"] * df["E(rk)"]

df["𝞼p"] = np.sqrt(   (df["wi"] ** 2) * (df["𝞼(ri)"] ** 2) 
                    + (df["wj"] ** 2) * (df["𝞼(rj)"] ** 2)
                    + (df["wk"] ** 2) * (df["𝞼(rk)"] ** 2) 
                    + 2 * df["wi"] * df["wj"] * df["Cov(i,j)"]
                    + 2 * df["wi"] * df["wk"] * df["Cov(i,k)"]
                    + 2 * df["wj"] * df["wk"] * df["Cov(j,k)"]  )

df["Sharpe ratio"] = (df["E(rp)"] - raito["rf"]["E(r)"]) / df["𝞼p"]


# ### Check out the maximum combination after have new weights

# In[32]:


df[df["Sharpe ratio"] == df["Sharpe ratio"].max()]


# In[33]:


df.to_excel("Final_Data.xlsx")


# In[ ]:




