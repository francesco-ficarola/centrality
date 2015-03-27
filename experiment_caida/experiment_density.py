import collections, sys, time, os.path, glob, pickle
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from copy import deepcopy

GRAPHS_DIR = '../datasets/as-caida/'

DATA_PATH = 'data/'
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
	
	print "\nAdding all nodes to the graph..."
	G = nx.DiGraph()
	for graph_file in data_files:
		print "Processing " + graph_file
		addNodesToGraph(G, graph_file)
	
	dict_single_graphs = {}
	total_all_edges = 0
	
	
	for iteration, graph_file in enumerate(data_files):
	
		print "\n\nITERATION: " + str(iteration) + ", processing " + graph_file
		buildGraph(G, graph_file)
		print "Nodes:", len(G)
		print "Edges:", len(G.edges())
		print "Dangling nodes (original graph):", str(countDanglingNodes(G))
		
		total_all_edges += len(G.edges())
	
	total_distinct_edges = len(G) * (len(G) - 1)
	total_graphs = len(data_files)
	print "\n\nTotal number of distinct edges: " + str(total_distinct_edges)
	print "Total number of all edges: " + str(total_all_edges)
	print "Total number of graphs: " + str(total_graphs)
	print "Density: " + str(float(total_all_edges) / float(total_distinct_edges * total_graphs))
		
	
if __name__ == '__main__':
	main()
