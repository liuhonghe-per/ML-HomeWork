#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import math


# In[12]:


def MSE(y,y_pre):
    error = y - y_pre
    sqr = error.dot(error)
    MSE_loss = np.sum(sqr)/np.size(sqr)
    return MSE_loss


# In[5]:


def RMSE(y,y_pre):
    error = y - y_pre
    sqr = error.dot(error)
    RMSE_loss = math.sqrt(np.sum(sqr)/np.size(sqr))
    return RMSE_loss


# In[6]:


def MAE(y,y_pre):
    error = y-y_pre
    error_abs = abs(error)
    MAE_loss = np.sum(error_abs)/np.size(error_abs)
    return MAE_loss


# In[14]:


def R_squared(y,y_pre):
    y_mean = np.mean(y)
    SSR = np.sum(np.square(y_pre-y_mean))
    SST = np.sum(np.square(y - y_mean))
    R_loss = SSR/SST
    return R_loss


# In[15]:


if __name__ == '__main__':
    y = np.array([1,2,5,8])
    y_pre = np.array([2,2,6,7])
    print(MSE(y,y_pre))
    print(RMSE(y,y_pre))
    print(MAE(y,y_pre))
    print(R_squared(y,y_pre))


# In[ ]:




