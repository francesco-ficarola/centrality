import collections, sys, time, os.path, glob, pickle, math
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr

GRAPH_FULL_FILE = '../datasets/facebook/facebook_graph.csv'
GRAPHS_DIR = '../datasets/facebook/singles/'
SNAPSHOT_PERIOD = 86400 #seconds = 1 day
ALPHA = 0.85
DELTA = 200
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


def computeCorrelation(curr_dict_rank, prev_dict_rank):
	curr_dict_rank = collections.OrderedDict(sorted(curr_dict_rank.items(), key=lambda t: t[0], reverse=False))
	prev_dict_rank = collections.OrderedDict(sorted(prev_dict_rank.items(), key=lambda t: t[0], reverse=False))
	curr_list_rank = curr_dict_rank.values()
	prev_list_rank = prev_dict_rank.values()
	return spearmanr(curr_list_rank, prev_list_rank)[0]


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
				if relationship == '1':
					G.add_edge(node1, node2)
				elif relationship == '-1':
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


def splittingFullFile(graph_file):
	data_files = glob.glob(GRAPHS_DIR + "*")
	if len(data_files) > 0:
		return
	
	print "Opening " + graph_file + "..."
	f = open(graph_file, 'r')
	lines = f.readlines()
	f.close()
	
	print "Splitting..."
	lastSnapshot = 0
	graphs = {}
	for line in lines:
		if not(line.startswith('#')):
			(timestamp, node1, node2, relationship) = line.strip().split(',')
#			print "Current timestamp: " + timestamp
			
			ts = long(timestamp)
			
			# First assignment
			if lastSnapshot == 0:
				lastSnapshot = ts
			
			# Current timestamp smaller than next snapshot
			if ts < (lastSnapshot + SNAPSHOT_PERIOD):
				if not lastSnapshot in graphs:
					graphs[lastSnapshot] = []
				
				if not [node1, node2, relationship] in graphs[lastSnapshot]:
					graphs[lastSnapshot].append([node1, node2, relationship])
					
			# Current timestamp greater than / equal to next snapshot
			else:
				# Increment lastSnapshot until timestamp will not be in the right snapshot (case of inexistent snapshots)
				lastSnapshot += SNAPSHOT_PERIOD
				graphs[lastSnapshot] = []
				
				while((lastSnapshot + SNAPSHOT_PERIOD) < ts):
					lastSnapshot += SNAPSHOT_PERIOD
					graphs[lastSnapshot] = []
#					print "Snapshot " + str(lastSnapshot) + ": " + str(graphs[lastSnapshot])
				
				graphs[lastSnapshot].append([node1, node2, relationship])
			
#			print "Snapshot " + str(lastSnapshot) + ": " + str(graphs[lastSnapshot])

	print "Writing splitted files..."
	for graph_ts in graphs:
		if len(graphs[graph_ts]) > 0:
			graph_edges_list = graphs[graph_ts]
			f_out = open(GRAPHS_DIR + str(graph_ts) + '.txt', 'w')
			for edge in graph_edges_list:
				f_out.write(str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(edge[2]) + '\n')
			f_out.close()


def main():

	print "\n\nSplitting the full graph in separated files..."
	splittingFullFile(GRAPH_FULL_FILE)

	data_files = sorted(glob.glob(GRAPHS_DIR + "*.txt"))
	if len(data_files) == 0:
		return
	
	print "\nAdding all nodes to the graph..."
	G = nx.DiGraph()
	for graph_file in data_files:
		print "Processing " + graph_file
		addNodesToGraph(G, graph_file)
	
	dict_single_graphs = {}
	
	print "\nLinking all graph files to a dictionary..."
	for iteration, graph_file in enumerate(data_files):
		dict_single_graphs[iteration] = graph_file
		
	f = open(OUTCOMES_PATH + 'correlations_pr_uar.csv', 'w')
	
	for delta in range(1, DELTA+1):
		print "DELTA: " + str(delta)
		all_k = []
		len_all_graphs = len(data_files)
		
		k1 = 0
		k2 = 0
		
		current_correlations_delta = []
		
		for simulation in range(NUMBER_OF_RANDOM_WINDOWS):
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
			
			dict_rank_PR_k1 = nx.pagerank(G_aggregated_k1, alpha=ALPHA, max_iter=ITERATIONS_PR, weight='weight')
			dict_rank_PR_k2 = nx.pagerank(G_aggregated_k2, alpha=ALPHA, max_iter=ITERATIONS_PR, weight='weight')
			
			corr = computeCorrelation(dict_rank_PR_k1, dict_rank_PR_k2)
			if not math.isnan(corr):
				current_correlations_delta.append(corr)
				
		avg_correlations = np.mean(current_correlations_delta)
		f.write(str(avg_correlations) + '\n')
		f.flush()
	
	f.close()
	
	
if __name__ == '__main__':
	main()
