import numpy as np
import scipy as sc
import pandas as pd
from fractions import Fraction
def display_format(my_vector, my_decimal):
   return np.round((my_vector).astype(np.float), decimals=my_decimal)
initial_rank = Fraction(1,5)
Mat = np.matrix([[0,0,1,0,0],
        [Fraction(1,2),0,0,0,0],
        [0,1,Fraction(1,3),0,1],
        [1,1,0,Fraction(1,4),0],
        [0,1,0,0,Fraction(1,4)],
        ])
temp = np.zeros((5,5))
temp[:] = initial_rank
beta = 0.6
Al = beta * Mat + ((1-beta) * temp)
r = np.matrix([initial_rank, initial_rank, initial_rank,initial_rank,initial_rank])
r = np.transpose(r)
previous_r = r
file = open("result.txt","w")
for i in range(1,50):
   r = Al * r
   file.write("Iteration:"+str(i)+"->"+str(display_format(r,4))+"\n")
   if (previous_r==r).all():
      break
   previous_r = r

file.write("Final ->"+str(display_format(r,4))+"\n")
file.close()