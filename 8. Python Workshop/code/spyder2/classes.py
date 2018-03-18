# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Create the class
class Bag:
  def __init__(self):
    self.data = []
    
  def add(self, x):
    self.data.append(x)
    
  def addtwice(self,x):
    self.add(x)    
    self.add(x)

# Object Instant
l = Bag()
l.add('first')
l.addtwice('second')

print l.data

l.data2 = 23


    