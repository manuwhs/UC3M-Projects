# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" Prompt input """

#texto = raw_input("Introduce un numero: ")
#numero = int(texto)

""" Reading file input """

archivo = open("./datos.txt","r")
datos_archivo = archivo.read()
archivo.close()

""" Writing to file """

archivo = open("./datos_salida.txt","w")

for i in range(20):
    archivo.write(str(i) + " ")
    
archivo.close()
