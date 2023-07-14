# -*- coding: utf-8 -*-
"""
DSAD Assignment 2 - Group 166

"""

import sys

class PharmacyRun():

    def __init__(self, input_file, output_file):
        self.vertices = []
        self.house = 0; self.pharm1 = 0; self.pharm2 = 0
        self.invalid = 0
        
        input_lines = (line.rstrip() for line in input_file)
        for x in input_lines:
            
            if x.find("/") != -1: 
                a = x.split("/")
                a = [y.strip(' ') for y in a] 

        # for every valid entry in input file, i.e. source, destination and distance exists
        # add an entry into list of vertices if the entry doesn't already exists
        # This is done to obtain unique list of vertices
                if a[0] and a[1] and a[2]:
                    if a[0] not in self.vertices:
                        self.vertices.append(a[0])
                    if a[1] not in self.vertices:
                        self.vertices.append(a[1])
            else:
                
                if x[:13] == "Harsh's House":
                    self.house = x[15:]
                if x[:10] == "Pharmacy 1":
                    self.pharm1 = x[12:]
                if x[:10] == "Pharmacy 2":
                    self.pharm2 = x[12:]
        
        # Check if any of the required inputs is missing. Set 'invalid' variable
        if not self.house or not self.pharm1 or not self.pharm2:
            output_file.write("Error: Required inputs missing\n")
            output_file.write("Error: Either Harsh's house or one of the pharmacy not specified")
            self.invalid = 1
            return
        
        # Check if house node and pharmacy node in graph. Set 'invalid' variable
        if self.house not in self.vertices or self.pharm1 not in self.vertices or self.pharm2 not in self.vertices:
            output_file.write("Error: Incorrect input found\n")
            output_file.write("Error: Either Harsh's house or one of the pharmacy not in graph")
            self.invalid = 1
            return
            
        self.vertices.sort()
               
        self.V = len(self.vertices)
        
       
       # create empty nxn adjacency matrix representation of graph 
       # where n is the number of valid vertices read from input file
        self.graph = [[0 for column in range(self.V)]
                      for row in range(self.V)]
        
        ''' This function creates individual entries in the adjacency matrix
        The index is determined is taken as per the sorted order of the vertices '''
    def createGraphEntry(self,v1,v2,weight):
        v1_index = self.vertices.index(v1)
        v2_index = self.vertices.index(v2)      
        self.graph[v1_index][v2_index] = int(weight)
    
    
    def writeResult(self, dest, path, zones):
        stack = []
        stack.append(dest)
        last_node = dest
        while last_node != -1:
            stack.append(path[last_node])
            last_node = path[last_node]
        
        output_file.write("\nPath to follow: ")
        for n in range(len(stack)-2,-1,-1):
            output_file.write(self.vertices[stack[n]] + ' ')
        
        output_file.write("\nContainment zones in this path: " + str(zones))
        
# This function takes care of the situation when paths to both 
# Pharmacies have equal number of containment zones        
        
    def writeResultMultiple(self, dest1, path1, dest2, path2, zones):
        stack = []
        stack.append(dest1)
        last_node = dest1
        while last_node != -1:
            stack.append(path1[last_node])
            last_node = path1[last_node]
        
        output_file.write("\nPath to follow for Pharmacy 1: ")
        for n in range(len(stack)-2,-1,-1):
            output_file.write(self.vertices[stack[n]] + ' ')
            
        stack = []
        stack.append(dest2)
        last_node = dest2
        while last_node != -1:
            stack.append(path2[last_node])
            last_node = path2[last_node]
        
        output_file.write("\nPath to follow for Pharmacy 2: ")
        for n in range(len(stack)-2,-1,-1):
            output_file.write(self.vertices[stack[n]] + ' ')
        
        output_file.write("\nContainment zones in both paths: " + str(zones))
 
    ''' This function finds the vertex with minimum edge weight, 
    from the set of vertices not yet present in shortest path tree '''
    def minWeight(self, weights, spt):
 
        min = sys.maxsize
        # initialize min_index to -1, it will remain as -1
        # if there is unreachable node as dist[v] will be max
        min_index = -1
 
        for v in range(self.V):
            if weights[v] < min and spt[v] == False:
                min = weights[v]
                min_index = v
 
        return min_index
 
    ''' This funtion follows Dijkstra's shortest path algorithm 
    for a graph in adjacency matrix representation '''
    def shortestPath(self, src, dest):
 
        # create a list of size equal to number of vertices
        # set all values to max int in the beginning except for source vertex
        weights = [sys.maxsize] * self.V
        weights[src] = 0
        
        # initialize shortest path tree 'spt' with all values as False
        spt = [False] * self.V
        shortestPath = [None] * self.V
        shortestPath[0] = -1
        self.path_found = 0
 
        for y in range(self.V):
 
            # Pick the vertex with minimum weight from the set of 
            # vertices not yet processed.
            u = self.minWeight(weights, spt)
            
            # u will be returned as -1 if the destination node is not reachable
            if u == -1:
                break
 
            # Add the vertex with min weight to the shotest path tree
            spt[u] = True
            
            # if destination vertex is reached, terminate the loop
            if u == dest:
                self.path_found = 1
                break
            
            # Update weigths of the adjacent vertices if the
            # vertex in not in the shotest path tree already
            for v in range(self.V):
                if self.graph[u][v] > 0 and spt[v] == False and weights[v] > weights[u] + self.graph[u][v]:
                    weights[v] = weights[u] + self.graph[u][v]
                    shortestPath[v] = u
        if self.path_found == 1:
            return(weights[dest], shortestPath)
        else:
            return(sys.maxsize,-1)

 
    
""" Main Program begins """

""" Initialize variables """

input_file_name = "inputPS14.txt" 
output_file_name = "outputPS14.txt"

input_file = open(input_file_name, "r")
output_file = open(output_file_name,"w")


g = PharmacyRun(input_file, output_file)

# Exit program if one of the required inputs is missing
if g.invalid == 1:
    sys.exit()

""" Re-open input file and read each line. Strip trailing newline character
    Read all lines that has / and Split using / as separator, remove leading and trailing whitespaces """

input_file = open(input_file_name, "r")

input_lines = (line.rstrip() for line in input_file)
for x in input_lines:
    if x.find("/") != -1: 
        a = x.split("/")
        a = [y.strip(' ') for y in a] 
        if a[0] and a[1] and a[2]:  
            g.createGraphEntry(a[0],a[1],a[2])

house_index = g.vertices.index(g.house)
pharm1_index = g.vertices.index(g.pharm1)
pharm2_index = g.vertices.index(g.pharm2)

pharm1_dist, path1 = g.shortestPath(house_index, pharm1_index)
pharm2_dist, path2 = g.shortestPath(house_index, pharm2_index)

if pharm1_dist == sys.maxsize and pharm2_dist == sys.maxsize:
    output_file.write("Error: Path not found to both Pharmacies")
    sys.exit()
if pharm1_dist != sys.maxsize and pharm2_dist == sys.maxsize:
    output_file.write("Path not found to Pharmacy 2\n")
if pharm1_dist == sys.maxsize and pharm2_dist != sys.maxsize:
    output_file.write("Path not found to Pharmacy 1\n")

    
dest1 = 0; dest2 = 0

if pharm1_dist <  pharm2_dist:
    output_file.write("Safer Pharmacy is: Pharmacy 1")
    dest1 = pharm1_index
    zones = pharm1_dist
    g.writeResult(dest1, path1, zones)
if pharm1_dist > pharm2_dist:
    output_file.write("Safer Pharmacy is: Pharmacy 2")
    dest2 = pharm2_index
    zones = pharm2_dist
    g.writeResult(dest2, path2, zones)
if pharm1_dist == pharm2_dist:
    # covers the case where the shortest distance is same 
    output_file.write("Both Pharmacies are equally safe")
    dest1 = pharm1_index
    dest2 = pharm2_index
    zones = pharm1_dist
    g.writeResultMultiple(dest1, path1, dest2, path2, zones)

  # Close input and output file
input_file.close()    
output_file.close()




































