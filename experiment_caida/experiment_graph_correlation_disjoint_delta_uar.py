import collections, sys, time, os.path, glob, pickle, math
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr

GRAPHS_DIR = '../datasets/as-caida/'
ALPHA = 0.85
DELTA = 60
ITERATIONS_PR = 100
NUMBER_OF_RANDOM_WINDOWS = 100

OUTCOMES_PATH = 'outcomes/delta_disjoint/'


def countDanglingNodes(G):
	count_dangling_nodes = 0
	for id_node in G:
		if len(G.successors(id_node)) == 0:
			count_dangling_nodes += 1
	return count_dangling_nodes


def totalElements(G, elemKey):
	total_elems = 0
	for id_node in G:
		total_elems += G.node[id_node][elemKey]
	return total_elems


def computeCorrelation(k1_dict, k2_dict):
	k1_dict = collections.OrderedDict(sorted(k1_dict.items(), key=lambda t: t[0], reverse=False))
	k2_dict = collections.OrderedDict(sorted(k2_dict.items(), key=lambda t: t[0], reverse=False))
	k1_list = k1_dict.values()
	k2_list = k2_dict.values()
	return pearsonr(k1_list, k2_list)[0]


def exportData(list_data, filename):
	f = open(OUTCOMES_PATH + filename, 'w')
	for elem in list_data:
		f.write(str(elem) + '\n')
	f.close()


def addNodesToGraph(G, graph_file):
	f = open(graph_file, 'r')
	lines = f.readlines()
	f.close()
	
	for line in lines:
		if not(line.startswith('#')):
			(node1, node2, relationship) = line.strip().split('\t')
			if not G.has_node(node1):
				G.add_node(node1)
			if not G.has_node(node2):
				G.add_node(node2)


def storeAllEdges(data_files):
	all_edges = {}
	for graph_file in data_files:
	
		f = open(graph_file, 'r')
		lines = f.readlines()
		f.close()
		
		for line in lines:
			if not(line.startswith('#')):
				(node1, node2, relationship) = line.strip().split('\t')
				if node1 != node2:
					if relationship == '-1':
						edge = (node1, node2)
						all_edges[edge] = 0.0
					elif relationship == '1':
						edge = (node2, node1)
						all_edges[edge] = 0.0
					elif relationship == '0' or relationship == '2':
						edge1 = (node1, node2)
						edge2 = (node2, node1)
						all_edges[edge1] = 0.0
						all_edges[edge2] = 0.0
	return all_edges


def buildGraph(G, graph_file):
	for edge in G.edges():
		G.remove_edge(*edge)
	
	f = open(graph_file, 'r')
	lines = f.readlines()
	f.close()
	
	for line in lines:
		if not(line.startswith('#')):
			(node1, node2, relationship) = line.strip().split('\t')
			if node1 != node2:
				if relationship == '-1':
					G.add_edge(node1, node2)
				elif relationship == '1':
					G.add_edge(node2, node1)
				elif relationship == '0' or relationship == '2':
					G.add_edge(node1, node2)
					G.add_edge(node2, node1)


def buildAggregatedGraph(delta_graphs):
	w = float(1) / float(len(delta_graphs))
	G_aggregated = nx.DiGraph()
	for graph in delta_graphs:
		for id_node in graph:
			if not G_aggregated.has_node(id_node):
				G_aggregated.add_node(id_node)
		for edge in graph.edges():
			if not G_aggregated.has_edge(*edge):
				G_aggregated.add_edge(*edge, weight=w)
			else:
				(src, trg) = edge
				G_aggregated[src][trg]['weight'] += w
	return G_aggregated


def main():

	data_files = sorted(glob.glob(GRAPHS_DIR + "*.txt"))
	if len(data_files) == 0:
		return
		
	print "\nSaving all edges in a dictionary..."
	all_edges = storeAllEdges(data_files)
	
	print "\nAdding all nodes to the graph..."
	G = nx.DiGraph()
	for graph_file in data_files:
		print "Processing " + graph_file
		addNodesToGraph(G, graph_file)
	
	dict_single_graphs = {}
	
	
	print "\nLinking all graph files to a dictionary..."
	for iteration, graph_file in enumerate(data_files):
		dict_single_graphs[iteration] = graph_file
		
	f = open(OUTCOMES_PATH + 'correlations_graphs_uar.csv', 'w')
	
	for delta in range(1, DELTA+1):
		print "\n\nDELTA: " + str(delta)
		all_k = []
		len_all_graphs = len(data_files)
		
		k1 = 0
		k2 = 0
		
		current_correlations_delta = []
		
		for simulation in range(NUMBER_OF_RANDOM_WINDOWS):	
			print "\nsimulation: " + str(simulation)		
			while(True):
				k1 = rnd.randint(0, len_all_graphs-1)
				k2 = k1 + delta
			
				if(not [k1, k2] in all_k and not [k2, k1] in all_k and (k2 + delta - 1) < len_all_graphs):
					all_k.append([k1, k2])
					all_k.append([k2, k1])
					break
			
			k1_delta_graphs = []
			k2_delta_graphs = []
		
			counter_graphs = 0
			while(counter_graphs < delta):
				buildGraph(G, dict_single_graphs[k1 + counter_graphs])
				k1_delta_graphs.append(G.to_directed())
				buildGraph(G, dict_single_graphs[k2 + counter_graphs])
				k2_delta_graphs.append(G.to_directed())
				counter_graphs += 1
				
			G_aggregated_k1 = buildAggregatedGraph(k1_delta_graphs)
			G_aggregated_k2 = buildAggregatedGraph(k2_delta_graphs)
			
			k1_edges = deepcopy(all_edges)
			k2_edges = deepcopy(all_edges)
			
			for edge in G_aggregated_k1.edges():
				(src, dst) = edge
				k1_edges[edge] = G_aggregated_k1[src][dst]['weight']
			
			print "Max weight k1_edges: " + str(max(k1_edges.values()))
				
			for edge in G_aggregated_k2.edges():
				(src, dst) = edge
				k2_edges[edge] = G_aggregated_k2[src][dst]['weight']
				
			print "Max weight k2_edges: " + str(max(k2_edges.values()))
			
			corr = computeCorrelation(k1_edges, k2_edges)
			print "Correlation: " + str(corr)
			if not math.isnan(corr):
				current_correlations_delta.append(corr)
				
		avg_correlations = np.mean(current_correlations_delta)
		f.write(str(avg_correlations) + '\n')
		f.flush()
	
	f.close()
	
	
if __name__ == '__main__':
	main()
