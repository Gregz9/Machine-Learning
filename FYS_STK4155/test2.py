import numpy as np 

a = np.array((1, 2, 3))

b = np.array(([1.0 , 1.0 , 1.0], 
              [1.0 , 1.0 , 1.0], 
              [1.0 , 1.0 , 1.0]))

c = a*b
print(c)
c = b@a
print(c)
