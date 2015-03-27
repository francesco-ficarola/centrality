import glob, collections, pickle, math
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

GRAPHS_DIR = '../datasets/as-caida/'
OUTCOMES_PATH = 'outcomes/'


def countDanglingNodes(G):
	count_dangling_nodes = 0
	for id_node in G:
		if len(G.successors(id_node)) == 0:
			count_dangling_nodes += 1
	return count_dangling_nodes
			

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

def main():

	data_files = sorted(glob.glob(GRAPHS_DIR + "*.txt"))
	if len(data_files) == 0:
		return
	
	print "\n\nAdding all nodes to the graph..."
	G = nx.DiGraph()
	for graph_file in data_files:
		print "Processing " + graph_file
		addNodesToGraph(G, graph_file)
	
	pageranks = []
	for iteration, graph_file in enumerate(data_files):
		print "\n\nITERATION: " + str(iteration) + ", processing " + graph_file
		buildGraph(G, graph_file)
		print "Nodes:", len(G)
		print "Edges:", len(G.edges())
		
		print "Dangling nodes (original graph):", str(countDanglingNodes(G))
		
		print "\nComputing the PageRank..."
		dict_rank_PR = nx.pagerank(G, max_iter=100, weight='weight')
		dict_rank_PR = collections.OrderedDict(sorted(dict_rank_PR.items(), key=lambda t: t[0], reverse=False))
		pageranks.append(dict_rank_PR)
	
	M = []
	print "\nComputing the correlation Matrix..."
	for i in range(len(data_files)):
		print '\nt=' + str(i)
		l = []
		for j in range(len(data_files)):
			corr = spearmanr(pageranks[i].values(), pageranks[j].values())[0]
			distance = 1
			if not math.isnan(corr):
				distance = math.sqrt(1 - math.pow(corr, 2))
				#distance = math.sqrt((1 - corr)/2.0)
			l.append(distance)
		M.append(l)
	print M
	
	f = open(OUTCOMES_PATH + 'correlations_matrix', 'w')
	pickle.dump(M, f)
	
	
if __name__ == '__main__':
	main()
