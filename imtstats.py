# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:08:03 2023

@author: miku_
"""
from scipy. stats import norm
import matplotlib.pyplot as plt
import statistics as stats
from pathlib import Path
import numpy as np

x = np.arange (12, 24, 0.001)
'''Дисперсия и Среднеквадратичное отклонение'''
def disp(i_m):
    gauss = []
    summ = []
    n = num()
    u = (sum(i_m))/n
    for i in i_m:
        summ.append((i - u)**2)
    S = np.sqrt(sum(summ)/n-1)
    print('стд. откл =', S)
    D = ((sum(summ))/n-1)
    print('Дисперсия=',D, '\n', 'мат оджидание =', u)
    for j in i_m:
        gauss.append((1/(S*(np.sqrt(2*np.pi))))*(np.exp(-(((j-u)**2)/(2*D)))))
    return gauss

'''ИМТ'''
def imt(height, weight):
    i_m = []
    for i, n in zip (height, weight):
        i_m.append(round(n/(i**2),2))
    return i_m

'''мода ряда'''
def range_fashion(i_m):
    m = []
    for i in i_m:
        m.append(i_m.count(i))
    return i_m[m.index(max(m))]

''''среднее арифметическое'''
def mean(i_m):
    count = 0
    for i in i_m:
        count += 1
        n = sum(i_m)/count
    return(n)

'''получить массивы из данныз в файле'''
def get_weight_from_data(weight):
    weight = []
    for i in wt:
        weight.append(float(i))
    return weight

def get_height_from_data(height):
    height = []
    for i in ht:
        height.append(float(i))
    return height

'''медиана'''
def med(i_m):
    i_m.sort()
    m = stats.median(i_m)
    return(m)
'''Выборка'''
def num():
    count = 0
    for i in weight:
        count += 1
    return count

'''ИМТ из таблицы'''
hh = [[20,21.3,22.7,24],[19.5,20.8,22.1,23.4,24.7],
      [19,20.2,21.5,22.8,24],[18.5,19.7,21,22.2,23.4,24.7],
      [19.2,20.4,21.6,22.8,24],[18.8,19.9,21.1,22.3,23.4,24.6],
      [19.4,20.6,21.7,22.9,24],[19,20.1,21.2,22.3,23.4,24.5],
      [18.5,19.6,20.7,21.8,22.9,24,25],[19.1,20.2,21.3,22.3,23.4,24.4],
      [18.7,19.7,20.8,21.8,22.8,23.9,24.9],[19.3,20.3,21.3,22.3,23.3,24.3],
      [18.8,19.8,20.8,21.8,22.8,23.8,24.8],[19.4,20.3,21.3,22.3,23.2,24.2],
      [18.9,19.9,20.8,21.8,22.7,23.7,24.6],[18.5,19.4,20.4,21.3,22.2,23.1,24.1,25],
      [19,19.9,20.8,21.7,22.6,23.5,24.5],[18.6,19.5,20.4,21.3,22.2,23,23.9,24.8],
      [19.1,19.9,20.8,21.7,22.5,23.4,24.3],[18.7,19.5,20.4,21.2,22.1,22.9,23.8,24.6],[19.1,19.9,20.8,21.6,22.4,23.3,24.1,24.9]]

d_e_imt = [18,17.6,17.1,16.7,15.9,15.6,
            15.2,14.9,14.5,14.2,13.9,13.6,
            13.3,13,12.7,12.5]
d_e_h = [1.58,1.60,1.62,1.64,1.66,1.68,
          1.70,1.72,1.74,1.76,1.80,1.82,
          1.84,1.86,1.88,1.90]
'''НОРМА'''
h = [1.50,1.52,1.54,1.56,1.58,1.60,1.62,
     1.64,1.66,1.68,1.70,1.72,1.74,1.76,1.78,
     1.80,1.82,1.84,1.86,1.88,1.90]
w = []
j = []
for i in hh:
    w.append(min(i))
    j.append(max(i))
    

'''ДАННЫЕ'''
ht = list(Path(r'путь к файлу с расширением .txt').read_text(encoding="utf-8").replace("\n", " ").split())
wt = list(Path(r'путь к файлу с расширением .txt').read_text(encoding="utf-8").replace("\n", " ").split())

weight = get_weight_from_data(wt)
height = get_height_from_data(ht)


'''GRAPH'''
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)

'''Данные'''
ax.plot((imt(height, weight)), height, 'ko')
'''#########Норма#нижняя###'''
ax.plot(w, h, 'bo')
'''#########Норма#верхняя###'''
ax.plot(j, h, 'bo')
'''#########Дефицит#нижняя###'''
ax.plot(d_e_imt, d_e_h, 'r-')

ax.plot(17.445, 1.57, 'mo')
plt.xlabel('ИМТ') 
plt.ylabel('Рост, м')

'''#########Данные#########'''
plt.scatter((imt(height, weight)), height)
'''#########Норма#низ########'''
plt.scatter(w, h)
'''#########Норма#верх########'''
plt.scatter(j,h)



plt.scatter(17.445, 1.57)
plt.minorticks_on()
plt.plot((imt(height, weight)), height, 'ko', label='Данные')
plt.plot(w, h, 'go', label='Нижняя граница нормы')
plt.plot(j, h, 'ro', label='Верхняя граница нормы')
plt.plot(d_e_imt, d_e_h, 'r-', label='Нижняя граница дефицит')
plt.legend(bbox_to_anchor=(0.48, -0.1))
plt.grid(which='major')
plt.grid(which='minor', linestyle=':') 
plt.show()

rak = imt(height, weight)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rak, disp(rak), 'ko')
plt.xlabel('ИМТ')
plt.ylabel('gauss')
plt.scatter(rak, disp(rak))

plt.plot (x, norm. pdf (x, 17.001358024691353, 2.208435988158504), label='μ:17.0013, σ: 2.2084')


plt.legend() 
plt.show()





print('мода ряда вес, kg =', range_fashion(weight),'\n'
      'мода ряда рост, m =', range_fashion(height),'\n'
      'мода ряда ИМТ =', range_fashion((imt(height, weight))),'\n'
      '\n'
      'среднее арифметическое вес, kg =', round (mean(weight),2),'\n'
      'среднее арифметическое рост, m =', round (mean(height),2),'\n'
      'среднее арифметическое ИМТ =', round (mean((imt(height, weight))),2),'\n'
      '\n'
      'медиана вес, kg =', med(weight),'\n'
      'медиана рост, m =', med(height),'\n'
      'медиана ИMТ =', med((imt(height, weight))),'\n'
      'Участников =', num())
#print(disp(rak))