# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:33:12 2015

@author: montoya
"""
# We use pandas library to read CSV data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import copy as copy

plt.close("all")
file_dir = "./[XAC] Correos.csv"
def read_dataset(file_dir = "./dataprices.csv"):
    
    data = pd.read_csv(file_dir, sep = ',',header = None, names = None) 
    # names = None coz the file does not contain column names
    Nsamples, Ndim = data.shape   # Get the number of bits and attr
    data_np = data
    
    return data_np

    
# Get the prices list and its shape
pipul = read_dataset(file_dir)


# There are people with 
Npeople, Ncol = pipul.shape

# Transformar en putas listas pork pandas es puta mierda !!!!!!!!
people = []
for i in range (Npeople):
    people.append([])
    for j in range (Ncol):
        people[i].append(pipul[j][i])


people = np.array(people).T

#### RANDOMIZE PEOPLE !!!!
order = np.random.permutation(Npeople)
#print people[0][16]
people = people.T[order].T

# Add coulumn with index numbers
identificador = range(Npeople)

people = np.insert(people,4,identificador,axis = 0)

#for i in range (Npeople):
#    people[i].append(i)
    
#print people[0][16]

# First column is name 
# Second is email
# Third is special character:
#   x -> As sparse as possible
#   y -> Active members. As sparse as possible
#   nan -> Dont care
#   Check point -> Does not count.
 
# Forth column has a code, people with the same code
# cannot go into the same group.
 

""" Que empiece el juego """ 

N_groups = 10

People_group = Npeople/N_groups
rem = Npeople % N_groups  # Number of groups with one more

Groups = []
for i in range(N_groups):
    Groups.append([])

# Get the PAX into lists and delete them from the total as they
# are being taken out.

# Copy the people

people_copy = copy.deepcopy(people)

# Get people that cannot be together
Non_tog_pos = np.where(people[3] == 'z0')[0]
Non_together = people[4][Non_tog_pos]
people = np.delete(people, Non_tog_pos, axis = 1)

# Get active members
Active_pos = np.where(people[2] == 'y')[0]
Active_members = people[4][Active_pos]
people = np.delete(people, Active_pos, axis = 1)

# Get Sparse members
Sparse_pos = np.where(people[2] == 'x')[0]
Sparse_members = people[4][Sparse_pos]
people = np.delete(people, Sparse_pos, axis = 1)

# Rest of members
Rest_PAX = people[4][:]

Nac = len(Active_members)
Nspar = len(Sparse_members)
Nnon = len(Non_together)
Nrest = len(Rest_PAX)

# Put the non_together:
aux = 0
for i in range(Nnon):
    aux = i % N_groups
    Groups[aux].append(Non_together[i])

# Put the members
prev = aux +1
for i in range(Nac):
    aux = (prev + i) % N_groups
    Groups[aux].append(Active_members[i])

prev = aux+1
# Put the sparse
for i in range(Nspar):
    aux = (prev + i) % N_groups
    Groups[aux].append(Sparse_members[i])

prev = aux+1
# Put the rest of the people
for i in range(Nrest):
    aux = (prev + i) % N_groups
    Groups[aux].append(Rest_PAX[i])
    
    


""" PRINT THE RESULTS """

N_rea = 0

text_file = open("./grupos"+str(N_rea)+".txt", "w")



for i in range(len(Groups)):
    text_file.write("Grupo "+str(i +1)+ "\n")
    
    # Randomize the order of print 
    order = np.random.permutation(len(Groups[i]))
    for j in range (len(Groups[i])):

        text_file.write(str(people_copy[0][int(Groups[i][order[j]])]) + "\n")
    text_file.write("\n")
    
text_file.close()