# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" if elif else """

age = 5
if  (age < 16) :
    print "too young"
    
elif  (age >= 35) :
    print "too old"
    
else :
    print "perfect match"

#if  (age < 16 or age >= 35) :
#    print "wrong match"
#    
#else :
#    print "perfect match"

""" while loop """
x = 0;
while (x < 10):
    x = x + 1
    if (x == 5):
        break
    print x

""" for loop """

#for i in range(0,10):
#    print x


lista = [1.4,"Hola mundo",89, [1,2,3]]
for elemento in lista:
    print elemento 
#
for i in range (len(lista)):
    print lista[i]




    