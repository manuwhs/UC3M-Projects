# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def print_suma(a ,b, lista):
    c = a + b
    a = 14
    
    lista[2] = 5
    
    print "La suma de " + str(a) + " + " + str(b) + " es " + str(c)
    
    return c

var = print_suma

listado = [1,2,3]
c = print_suma(3,10,listado)

var(5,4,listado)
#mb.print_suma("hola ", "mundo ")