#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[23]:


#Memanggil dataset
dataset = pd.read_csv("DataKepuasan.csv")
#Sumbu X adalah Kepuasan, dan Sumbu Y adalah Kunjungan Pelanggan
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[24]:


#Menampilkan isi sebagian dataset
dataku = pd.DataFrame(dataset)
#Visualisasi Data
plt.scatter(dataku.Kunjungan, dataku.Kepuasan)
plt.xlabel("Kungjungan")
plt.ylabel("Kepuasan")
plt.title("Grafik Kunjungan vs Kepuasan")
plt.show()


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[26]:


#Melakukan fitting simple linear regression pada training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[27]:


plt.figure(figsize=(10,8))
#Biru adalah data observasi
plt.scatter(X_train, y_train, color = 'blue')
#Garis merah adalah hasil prediksi dari machine learning
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title ('Jumlah Kunjungan terhadap Kepuasan')
plt.xlabel('Jumlah Kunjungan')
plt.ylabel('Kepuasan')
plt.show()


# In[28]:


#Biru adalah data observasi
plt.scatter(X_test, y_test, color = 'blue')
#Merah adalah hasil prediksi dari machine learning
plt.plot(X_train, regressor.predict(X_train), color = 'red')
#Judul dan label
plt.title ('Kepuasan vs Kunjungan (Testing set)')
plt.xlabel('Kunjungan')
plt.ylabel('Kepuasan')
plt.show()

