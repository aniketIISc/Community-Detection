import networkx as nx
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_facebook_data(file_path):
    nodes_connectivity_list_fb = []
    with open(file_path, 'r') as file:
        for line in file:
            first, second = line.strip().split()
            nodes_connectivity_list_fb.append([int(first), int(second)])
            nodes_connectivity_list_fb.append([int(second), int(first)])
    nodes_connectivity_list_fb = np.array(nodes_connectivity_list_fb)
    return nodes_connectivity_list_fb

def import_bitcoin_data(file_path):
  data = np.genfromtxt(file_path, delimiter=',', dtype=int)
  data[:,2]+=10
  data_wts = data[:,:-1]
  nodes_connectivity_list_btc = []
  for edge in data_wts:
      nodes_connectivity_list_btc.append([edge[0], edge[1], edge[2]])
      nodes_connectivity_list_btc.append([edge[1], edge[0], edge[2]])
  nodes_connectivity_list_btc  = np.array(nodes_connectivity_list_btc, dtype=int)
  return nodes_connectivity_list_btc

def getLaplacian(degree, adjacencyMatrix):
    return degree - adjacencyMatrix

def getAdjacency(edgeList, num_vertices):
    adjacencyMatrix = np.zeros((num_vertices, num_vertices))
    for edge in edgeList:
        source, dest = edge[0], edge[1]
        adjacencyMatrix[source, dest] += 0.5
        adjacencyMatrix[dest, source] += 0.5
    return adjacencyMatrix
    
def getDegreeMatrix(adjacencyMatrix, num_vertices):
    degree = np.zeros((num_vertices, num_vertices))
    for source in range(num_vertices):
        degree[source, source] = np.sum(adjacencyMatrix[source])
    return degree

def getFiedlerVector(laplacian, degree):
    evals, evec = np.linalg.eig(laplacian)
    indexes = np.argsort(evals)
    evals = evals[indexes]
    evec = evec[:, indexes]
    fiedlerVector = np.real(evec[:,1])
    indexes = np.argsort(fiedlerVector)
    return fiedlerVector, indexes

def getAdjacencyBTC(nodes_connectivity_list_btc):
  num_vertices = np.max(nodes_connectivity_list_btc)+1
  adjacency_matrix = np.zeros((num_vertices, num_vertices))
  for entry in nodes_connectivity_list_btc:
      source , dest, wt = entry[0], entry[1], entry[2]
      adjacency_matrix[source, dest] += 0.5*wt
      adjacency_matrix[dest, source] = 0.5*wt
  return adjacency_matrix

def getDegreeMatrixBTC(adjacency_matrix):
  num_vertices = adjacency_matrix.shape[0]
  degree = np.zeros((num_vertices, num_vertices))
  for source in range(num_vertices):
      degree[source, source] = np.sum(adjacency_matrix[source, :])
  return degree

def getSubAdjacencyMatrix(adjacencyMatrix, indexes):
    return adjacencyMatrix[np.ix_(indexes, indexes)]

def getPartition( fiedlerVector, indexes):
    for i in indexes:
        if(fiedlerVector[i]>0):
            return i

def split(friedelVector):
    n = friedelVector.shape[0] 
    if n < 100:
        return False
    maximum = 0
    sumgap = 0
    for i in range(friedelVector.shape[0]-1):
        gap = friedelVector[i+1] - friedelVector[i]
        sumgap += gap
        if gap >maximum:
            maximum = gap
    if maximum > 100*(sumgap/(n-1)):
        return True
    return False


def spectralDecomp_OneIter(nodes_connectivity_list_fb):
    num_vertices = np.max(nodes_connectivity_list_fb)+1
    adjacencyMatrix = getAdjacency(nodes_connectivity_list_fb, num_vertices)
    degree = getDegreeMatrix(adjacencyMatrix, num_vertices)
    L = getLaplacian(degree, adjacencyMatrix)
    fiedler_vector, indexes = getFiedlerVector(L, degree)
    adjacency_matrix_sorted = np.zeros((num_vertices, num_vertices))
    adjacency_matrix_sorted[range(num_vertices),:] = adjacencyMatrix[indexes,:]
    adjacency_matrix_sorted[:, range(num_vertices)] = adjacency_matrix_sorted[:,indexes]
    partition = np.zeros((num_vertices, 2))
    for i in range(len(fiedler_vector)):
        partition[i, 0] = i
        if(fiedler_vector[i] < 0):
            partition[i, 1] = 0
        else:
            partition[i, 1] = 1
    return np.sort(fiedler_vector), adjacency_matrix_sorted, partition

def spectralDecomposition(nodes_connectivity_list_fb):   
    num_vertices = np.max(nodes_connectivity_list_fb)+1
    adjacencyMatrix = getAdjacency(nodes_connectivity_list_fb, num_vertices)
    degree = getDegreeMatrix(adjacencyMatrix, num_vertices)
    laplacian = getLaplacian(degree, adjacencyMatrix)
    fiedlerVector, indexes = getFiedlerVector(laplacian, degree)
    part = getPartition(fiedlerVector, indexes)
    fiedlerVectorMapping = []
    fiedlerVectorSorted = np.sort(fiedlerVector)
    fiedlerVectorMapping.append((fiedlerVectorSorted[:part], indexes[:part]))
    fiedlerVectorMapping.append((fiedlerVectorSorted[part:], indexes[part:]))
    i = 0
    while(i< len(fiedlerVectorMapping)):
        community = fiedlerVectorMapping[i]
        if(split(community[0])):
            newAdjacencyMatrix = getSubAdjacencyMatrix(adjacencyMatrix, community[1])
            newDegree = getDegreeMatrix(newAdjacencyMatrix, len(community[0]))
            newLaplacian = getLaplacian(newAdjacencyMatrix, newDegree)
            newfiedlerVector, newIndexes = getFiedlerVector(newLaplacian, newDegree)
            newpart = getPartition(newfiedlerVector, newIndexes)
            newIndexes = np.array([community[1][k] for k in newIndexes])
            newfiedlerVector = np.sort(newfiedlerVector)
            minNodes = min(newfiedlerVector[:newpart].shape[0], newfiedlerVector[newpart:].shape[0])
            if(minNodes <=50):
              i+=1
              continue
            fiedlerVectorMapping[i] = (newfiedlerVector[:newpart], newIndexes[:newpart])
            fiedlerVectorMapping.insert(i+1, (newfiedlerVector[newpart:], newIndexes[newpart:]))
        else:
            i+=1
    partition = np.zeros((num_vertices, 2))
    c=0
    for l in fiedlerVectorMapping:
        community_nodes = l[1]
        for node in community_nodes:
            partition[node, 0] = int(node)
            partition[node, 1] = int(c)
        c+=1
    partition=partition.astype(int)
    return partition



def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
    num_nodes = graph_partition_fb.shape[0]
    adjacency_matrix_sorted = np.zeros((num_nodes, num_nodes))
    unique_values, value_counts = np.unique(graph_partition_fb[:, 1], return_counts=True)
    value_count_dict = {value: count for value, count in zip(unique_values, value_counts)}
    sorted_data = graph_partition_fb[np.argsort([value_count_dict[value] for value in graph_partition_fb[:, 1]])]
    sorted_data = np.array(sorted_data)
    indexes = sorted_data[:,0]
    adjacency_matrix = getAdjacency(nodes_connectivity_list_fb, num_nodes)
    adjacency_matrix_sorted = np.zeros((num_nodes, num_nodes))
    adjacency_matrix_sorted[range(num_nodes),:] = adjacency_matrix[indexes,:]
    adjacency_matrix_sorted[:,range(num_nodes)] = adjacency_matrix_sorted[:, indexes]
    return adjacency_matrix_sorted

class Louvain:
    def __init__(self, num_vertices, adjacency_matrix) -> None:
        self.community = np.arange(num_vertices)
        self.num_vertices = num_vertices
        self.degree = np.sum(adjacency_matrix, axis=1)
        self.adjacency_matrix = adjacency_matrix
        
        
    def demerge(self, i ):
        total_degree = np.sum(self.degree)
        comm = self.community[i]
        community_nodes = np.where(self.community==comm)[0]
        sigma_total = sum(self.degree[node] for node in community_nodes)
        k_i_out = 2*np.sum(self.adjacency_matrix[i, community_nodes])
        k_i = self.degree[i]
        Q = (((2*k_i*sigma_total) - (2*k_i**2))/total_degree  - (k_i_out))/total_degree
        return Q

    def merge(self, i, comm ):
        total_degree = np.sum(self.degree)
        community_nodes = np.where(self.community==comm)[0]
        sigma_total = sum(self.degree[node] for node in community_nodes)
        k_i_in = 2*np.sum(self.adjacency_matrix[i, community_nodes])
        k_i = self.degree[i]
        Q = (k_i_in - ((2*sigma_total*k_i)/total_degree))/total_degree
        return Q
    
    def calculate_modularity(self):
        community_indexs = np.unique(self.community)
        Q = 0
        total_degree = np.sum(self.degree)
        for ind in community_indexs:
            community_nodes = np.where(self.community == ind)[0]
            sigma_total = sum(self.degree[node] for node in community_nodes)
            sigma_in = np.sum(self.adjacency_matrix[np.ix_(community_nodes,community_nodes)])
            Q+= (sigma_in/total_degree - (sigma_total**2/total_degree**2) )   
        return Q
    
    def first_phase(self):
        flag = True
        while(flag):
            count = 0
            for i in range(self.num_vertices):
                neighborhood = np.unique(self.community[np.where(self.adjacency_matrix[i]!=0)[0]])
                community_i = self.community[i]
                Q_demerge = self.demerge(i)
                Q_max = 0
                for j in neighborhood:
                    if(j == self.community[i]):
                        continue
                    Q_merge = self.merge(i,j)
                    Q = Q_demerge + Q_merge
                    # print(Q)
                    if(Q_max < Q):
                        Q_max = Q
                        community_i = j
                if(Q_max > 0 and community_i != self.community[i]):
                    self.community[i] = community_i
                    count = count+1
            print(count,len(np.unique(self.community)),self.calculate_modularity())
            if(count == 0):
                break
        return self.community


def louvain_one_iter(nodes_connectivity_list_fb):
    num_vertices = np.max(nodes_connectivity_list_fb)+1
    adjacencyMatrix = getAdjacency(nodes_connectivity_list_fb, num_vertices)
    L = Louvain(num_vertices, adjacencyMatrix)
    graph_partition_louvain_fb = L.first_phase()
    return graph_partition_louvain_fb

    
if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    
    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    print(graph_partition_fb)
    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    print(graph_partition_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    print(graph_partition_louvain_fb)
    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
    print(graph_partition_btc)
    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    print(graph_partition_btc)
    
    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    print(graph_partition_louvain_btc)
