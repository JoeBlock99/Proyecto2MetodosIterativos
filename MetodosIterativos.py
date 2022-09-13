from pprint import pprint
from tkinter.messagebox import NO
import numpy as np
from tabulate import tabulate
import time
import random

def matrixGenerator(length):
    matrix = []
    result = []
    solution = []
    for i in range(length):
        row = []
        solution.append(0)
        result.append(random.randrange(-25, 25))
        for j in range(length):
            if i == j:
                row.append(random.randrange(10, 25))
            else:    
                row.append(random.randrange(-10, 10))
        matrix.append(row)
    return matrix, result, solution


def seidel(a, x ,b, n):                       
    for j in range(0, n):        
        d = b[j]                  
          
        # Para calcular xi, yi, zi
        for i in range(0, n):     
            if(j != i):
                d-=a[j][i] * x[i]      
        x[j] = d / a[j][j]
    return x                  
def gauss_seidel(A,b,N=25,x=None, tabla=None):
    print("Gauss Seidel:\n")
    n = len(A)   
    format_row = "{:>3}" + " |{:>30}, " + "{:>30}, "*(n-2) + "{:>30}|"
    format_header = "{:>3}" + "  {:>30}  " + "{:>30}  "*(n-2) + "{:>30} "
    header = []
    hline = []
    if tabla != None:
        for i in range(n):
            header.append('x_'+ str(i+1))
            hline.append('='*30)
        print(format_header.format("", *header))
        print(format_header.format("", *hline))
    start = time.perf_counter()
    for i in range(N):
        x = seidel(A, x, b, n)
        if tabla != None:
            print(format_row.format(i+1,*x))
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f"Duracion: {ms:.03f} micro secs.")
    return x 

###########JACOBI#############
def jacobi(A,b,N=25,x=None, tabla=None):
    print("Jacobi:\n")
    n = len(A)   
    format_row = "{:>3}" + " |{:>30}, " + "{:>30}, "*(n-2) + "{:>30}|"
    format_header = "{:>3}" + "  {:>30}  " + "{:>30}  "*(n-2) + "{:>30} "
    header = []
    hline = []
    if tabla != None:
        for i in range(n):
            header.append('x_'+ str(i+1))
            hline.append('='*30)
        print(format_header.format("", *header))
        print(format_header.format("", *hline))
    # Genera una posibilidad                                                                                                                                                           
    if x is None:
        x = np.zeros(len(A[0]))

    # Crea un vector a partir de la diagonal de A y se lo resta a A.
    start = time.perf_counter()                                                                                                                                                                 
    D = np.diag(A)
    R = A - np.diagflat(D)
                                                                                                                                                                         
    for i in range(N):
        x = (b - np.dot(R,x)) / D
        if tabla != None:
            print(format_row.format(i+1,*x))
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f"Duracion: {ms:.03f} micro secs.")
    return x

def EliminacionGaussiana(A,b,pasos='N'):
    print("Eliminacion Gaussiana:\n")
    m = np.shape(A)[0]  # número de filas
    n = np.shape(A)[1]  # número de columnas
    b_o = []
    for i in b:
        b_o.append([i])
    if m != n:
        print('La matriz debe ser cuadrada para realizar la eliminación gaussiana')
    start = time.perf_counter()
    Aug = np.append(A, b_o, 1)
    for k in range(0, n-1):
        for i in range(k + 1, n):
            factor = Aug[i][k] / Aug[k][k]
            Aug[i,k:n+1] = Aug[i,k:n+1] - factor * Aug[k,k:n+1]
        if pasos == 'S':
            print('Eliminación hacia adelante de la columna ', k + 1, '\n')
            np.set_printoptions(suppress=True)
            print(Aug)
            print(' ')
    x = np.zeros((n, 1))
    x[n - 1] = Aug[n - 1][n] / Aug[n - 1][n - 1]  # El vector del lado derecho es Aug(:, nb)
    for i in range(n - 2, -1, -1):
        x[i] = (Aug[i][n] - np.dot(Aug[i][i+1:n], x[i + 1:n])) / Aug[i][i]
    end = time.perf_counter()
    ms = (end-start) * 10**6
    print(f"Duracion: {ms:.03f} micro secs.")
    return x


##################Examples#################
# print("Ejemplo#1:\n")
x = [0, 0, 0]                        
# a = [[4, 1, 2],
#      [3, 5, 1],
#      [1, 1, 3]]
# b = [4,7,3]
# r = gauss_seidel(a, b, 30, x) 
# print('Resultado:\n',r,'\n\n')  
# r = jacobi(a, b, 30, x)
# print('Resultado:\n',r,'\n\n') 
# a = [[4, 1, 2],
#      [3, 5, 1],
#      [1, 1, 3]]
# b = [4,7,3]
# r = EliminacionGaussiana(a,b,'N')
# print('Resultado:\n',r,'\n\n')
# print("Ejemplo#2:\n")
# x = [0, 0, 0]    
# a = [[4.0, 2, 1],
#      [ 2 , 4, 2], 
#      [-1 , 2, 4]] 
# b = [12,12,8]
#A2 = np.array([[4.0, 2.0,-1.0],[ 2.0,1.5,-2.0], [-3.0,0.2,1.0]]) # matriz A
# r = gauss_seidel(a, b, 60, x) 
# print('Resultado:\n',r,'\n\n') 
# r = jacobi(a, b, 60, x)
# print('Resultado:\n',r,'\n\n')
# r = EliminacionGaussiana(a,b,'S')
# print('Resultado:\n',r,'\n\n')
# print("Ejemplo#3:\n")
# a = [[4, -1,-1],[-2,6,1], [-1,1,7]] # matriz A
# b = [3,9,-6]
# r = gauss_seidel(a, b, 30, x) 
# print('Resultado:\n',r,'\n\n') 
# r = jacobi(a, b, 30, x)
# print('Resultado:\n',r,'\n\n')
# r = EliminacionGaussiana(a,b,'N')
# print('Resultado:\n',r,'\n\n')

print("Ejemplo#3:\n")
x = [0,0,0,0,0, 0]
a = [[4, -1,-1, 2, -1,  3],
     [-2, 6, 1, 3, -2,  1], 
     [-1, 1, 7, 5,  4, -2],
     [-2, 3, 4, 8,  1, -4],
     [-3, 1, 3, 2,  6,  1],
     [ 1, 2, 3, 2,  1,  7]] 
b = [3,9,-6, 10, -4, 8]
r = gauss_seidel(a, b, 30, x) 
print('Resultado:\n',r,'\n\n') 
r = jacobi(a, b, 30, x)
print('Resultado:\n',r,'\n\n')
r = EliminacionGaussiana(a,b,'N')
print('Resultado:\n',r,'\n\n')
# print("Ejemplo#4:\n")
# a, b, x = matrixGenerator(10)
# print("a:\n")
# for i in a:
#     print(i)
# print("b:\n",b)
# print("x:\n",x)
# r = gauss_seidel(a, b, 100, x) 
# print('Resultado:\n',r,'\n\n') 
# r = jacobi(a, b, 100, x)
# print('Resultado:\n',r,'\n\n')
# r = EliminacionGaussiana(a,b,'N')
# print('Resultado:\n',r,'\n\n')