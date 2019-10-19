import numpy as np
import random
a=np.arange(0,10)
b=np.arange(10,20)
print(a,b)
randnum = random.randint(0,100)
print(randnum)
random.seed(randnum)
random.shuffle(a)
random.seed(randnum)
random.shuffle(b)
print(a,b)