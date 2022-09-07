import numpy as np
from tabulate import tabulate

def seidel(a, x ,b):      
    n = len(a)                   
    for j in range(0, n):        
        d = b[j]                  
          
        # Para calcular xi, yi, zi
        for i in range(0, n):     
            if(j != i):
                d-=a[j][i] * x[i]      
        x[j] = d / a[j][j]
    return x                  

def jacobi(A,b,N=25,x=None):
    # Genera una posibilidad                                                                                                                                                           
    if x is None:
        x = np.zeros(len(A[0]))

    # Crea un vector a partir de la diagonal de A y se lo resta a A.                                                                                                                                                                 
    D = np.diag(A)
    R = A - np.diagflat(D)
                                                                                                                                                                         
    for i in range(N):
        x = (b - np.dot(R,x)) / D
    return x
###########SEIDEL#############
x = [0, 0, 0]                        
a = [[4, 1, 2],[3, 5, 1],[1, 1, 3]]
b = [4,7,3]

result_index = []  
result_a = [0]
result_x = [0]
result_b = [0]
for i in range(0, 25):            
    x = seidel(a, x, b)
    result_index.append(i)
    result_a.append(x[0])
    result_x.append(x[1])
    result_b.append(x[2])
print(tabulate({'#:': result_index, 'a:': result_a, 'x:': result_x,  'b:': result_b}, headers="keys", tablefmt='fancy_grid'))     


###########JACOBI#############
A = np.array([[2.0,1.0],[5.0,7.0]])
b = np.array([11.0,13.0])
guess = np.array([1.0,1.0])

sol = jacobi(A,b,N=25,x=guess)

print("A:")
print(A)

print("b:")
print(b)

print("x:")
print(sol)