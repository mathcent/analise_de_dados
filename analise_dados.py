#!/usr/bin/env python
# coding: utf-8

# In[41]:


#----------------QUESTÕES ORDENADAS RESPECTIVAMENTE----------------

#Média, variância, desvio padrão e mediana para x e y.
#O histograma de x e y.
#O coeficiente de correlação de x e y.
#Fazer o teste de normalidade para  y e x.

#------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from scipy import stats
 
df = pd.read_csv('winequality-red.csv')
 
df.isnull().sum().sort_values(ascending = False)[:10]
print("Nº LINHAS / Nº COLUNAS: {}".format(df.shape))
attributes = list(df.columns)

df.dropna()
df.fillna(df.mean(0))
df.drop_duplicates()
df.head(25)

#------------------------------------------------------------------


# In[42]:


#------------------------------------------------------------------

df = df.to_numpy()
 
x = df[:, 0]
y = df[:, -1]
 
print(x)
print(y)

#------------------------------------------------------------------


# In[43]:


#----------------MÉDIA----------------

mediaX = np.mean(x)
mediaY = np.mean(y)

print("MÉDIA DE X: {} E MÉDIA DE Y: {}".format(str(mediaX), str(mediaY)))

#------------------------------------------------------------------


# In[44]:


#----------------VARIÂNCIA----------------

varianciaX = np.var(x) 
varianciaY = np.var(y) 

print("VARIÂNCIA DE X: {} E VARIÂNCIA DE Y: {}".format(str(varianciaX), str(varianciaY)))

#------------------------------------------------------------------


# In[45]:


#----------------DESVIOS PADRÃO----------------

dpx = np.std(x)
dpy = np.std(y)
 
print("DP DE X: {} E DP DE Y: {}".format(str(dpx), str(dpy)))

#------------------------------------------------------------------


# In[46]:


#----------------MEDIANA----------------

mx = np.median(x)
my = np.median(y)
 
print("MEDIANA DE X: {} E MEDIANA DE Y: {}".format(str(mx), str(my)))

#------------------------------------------------------------------


# In[40]:


#----------------HISTOGRAMA 1----------------

his1 = np.histogram(x, bins = 'auto')

print(his1)

plt.hist(x, bins = 'auto')
plt.title('HISTOGRAMA DE X')
plt.xlabel('VAL.')
plt.ylabel('FREQ.')
plt.show()

#------------------------------------------------------------------


# In[47]:


#----------------HISTOGRAMA 2----------------

his2 = np.histogram(y, bins = 'auto')

print(his2)

plt.hist(y, bins = 'auto')
plt.title('HISTOGRAMA DE Y')
plt.xlabel('VAL.')
plt.ylabel('FREQ.')
plt.show()

#------------------------------------------------------------------


# In[48]:


#----------------COEFICIENTE DE CORRELAÇÃO----------------

co = np.corrcoef(x, y)

print(co)

#------------------------------------------------------------------


# In[49]:


#----------------TESTE DE NORMALIDADE 1----------------

stat, p = shapiro(x)

print('EST. = %.3f, P = %.3f' % (stat, p))
 
alpha = 0.05

if p > alpha:
    print('AMOSTRA GAUSSIANA')
else:
    print('AMOSTRA NÃO GAUSSIANA')

plt.hist(x, bins = 'auto')
plt.title('TESTE DE NORMALIDADE DE X')
plt.ylabel('FREQ.')
plt.xlabel('VAL.')
plt.show()

#------------------------------------------------------------------


# In[50]:


#----------------TESTE DE NORMALIDADE 2----------------

stat, p = shapiro(y)

print('EST. = %.3f, P = %.3f' % (stat, p))
 
alpha = 0.05

if p > alpha:
    print('AMOSTRA GAUSSIANA')
else:
    print('AMOSTRA NÃO GAUSSIANA')

plt.hist(y, bins = 'auto')
plt.title('TESTE DE NORMALIDADE DE Y')
plt.ylabel('FREQ.')
plt.xlabel('VAL.')
plt.show()

#------------------------------------------------------------------


# In[ ]:



