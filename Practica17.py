# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 03:23:31 2023

@author: Lupita
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#DIEGO ALVIZO

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, : 1].values

#dividir la data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state= 0)

#crear modelo
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

#predecir el conjunto de test
y_pred = regression.predict(x_test)

#visualizar
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regression.predict(x_train), color = "blue")
plt.title("sueldo vs año  (Conjunto de experiencias)Diego Alvizo")
plt.xlabel("años de experiencia")
plt.ylabel("sueldo (en $)")
plt.show()

#resultados
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regression.predict(x_train), color = "blue")
plt.title("sueldo vs año  (Conjunto de experiencias)Diego Alvizo")
plt.xlabel("años de experiencia")
plt.ylabel("sueldo (en $)")
plt.show()

