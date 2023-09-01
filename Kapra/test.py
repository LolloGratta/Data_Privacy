from treelib import Node as Nodegraph, Tree
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from loguru import logger
from saxpy.paa import paa
dicto = dict()
dicta = dict()
dicto[2] = [2,3,4]
dicto[3] = [1,1]
dicto[4] = ([1,2],"ciao")
dicta[32] = [1]
dictum = dicto | dicta

lista = [1,2,3,4,5,6]
list_copia = lista.copy()
lista[0] = 5
print(list_copia)

data = np.array([44,34,33,39,34,30,47,27,45,39,47,39,35,47,29,45,52,38,51,38,41,21,33,33,25,27,32,42,26,33,26,26,32,31,26,38,46,28,30,31,41,21,40,33,32,26,26,29,44,30,34,28])
data_znorm = znorm(data)
data_paa = paa(data_znorm, 3)
pr = ts_to_string(data_paa, cuts_for_asize(5))

data = np.array([0,1,2])
data_znorm = znorm(data)
data_paa = paa(data_znorm,2)
pr = ts_to_string(data_paa,cuts_for_asize(2))
print("PR_ ",pr)
print("PAA: ",data_paa)

import sys
import time

# Start timer 

from loguru import logger
from datetime import timedelta


logger.remove(0)
logger.add(sys.stderr, colorize=True, format="[<g>{elapsed} | {level}: {message}</g> ] " )
logger.debug("CIAooooooooooo ")

def get_level(letter):
        if "a" <= letter < "t":
            return ord(letter) - 97
        
ciao = "ciao"
stronzo = "stronzo"
print(ciao+stronzo)

data = np.array([3,10,1,4])
data_znorm = znorm(data)
print(data_znorm)
print((data_znorm[0]+data_znorm[1])/2," , ", (data_znorm[2]+data_znorm[3])/2)
data_paa = paa(data_znorm, 2)
pr = ts_to_string(data_paa, cuts_for_asize(3))
print(pr)

pts = np.random.normal(size=10000)
prova = np.median(pts[(pts >= -0.43) & (pts < 0.43)])
print(prova)

from saxpy.strfunc import idx2letter

print(get_level("b"))




tupla = (1,2)

print(tupla[1])