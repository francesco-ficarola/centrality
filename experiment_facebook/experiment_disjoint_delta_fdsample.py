import collections, sys, time, os.path, glob, pickle
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from copy import deepcopy

GRAPH_FULL_FILE = '../datasets/facebook/facebook_graph.csv'
GRAPHS_DIR = '../datasets/facebook/singles/'
SNAPSHOT_PERIOD = 86400 #seconds = 1 day
ALPHA = 0.85
DELTA = 10
RATE = 1
ITERATIONS_PR = 100
PRECISION_K = [10, 20, 50, 100, 500, 1000]

DATA_PATH = 'data/'
OUTCOMES_PATH = 'outcomes/delta_disjoint/'
PAGERANK_DATA = ['dict_rank_PR']
FDSAMPLE_DATA = ['est_rank_FDS', 'tokens_FDS']


def checkExistingData(list_files, rate=-1, delta=-1, beta=-1, iterations=-1):
	data_files = glob.glob(DATA_PATH + "*")
	if len(data_files) == 0:
		return False
		
	append_string = ""
	if rate != -1:
		append_string += "_rate" + str(rate)
	if delta != -1:
		append_string += "_delta" + str(delta)
	if beta != -1:
		append_string += "_beta" + str(beta)
	if iterations != -1:
		append_string += "_iter" + str(iterations)
		
	for filename in list_files:
		total_filename = DATA_PATH + filename + append_string
		if total_filename not in data_files:
			return False
	return True


def getExistingData(list_files, rate=-1, delta=-1, beta=-1, iterations=-1):
	output_files = []
	
	append_string = ""
	if rate != -1:
		append_string += "_rate" + str(rate)
	if delta != -1:
		append_string += "_delta" + str(delta)
	if beta != -1:
		append_string += "_beta" + str(beta)
	if iterations != -1:
		append_string += "_iter" + str(iterations)
	
	for filename in list_files:
		total_filename = DATA_PATH + filename + append_string
		pkl_file = open(total_filename, 'rb')
		output_files.append(deepcopy(pickle.load(pkl_file)))
		pkl_file.close()
		
	return output_files


def writeData(dict_data, rate=-1, delta=-1, beta=-1, iterations=-1):
	append_string = ""
	if rate != -1:
		append_string += "_rate" + str(rate)
	if delta != -1:
		append_string += "_delta" + str(delta)
	if beta != -1:
		append_string += "_beta" + str(beta)
	if iterations != -1:
		append_string += "_iter" + str(iterations)
		
	for k, v in dict_data.items():
		total_filename = DATA_PATH + k + append_string
		output_file = open(total_filename, 'wb')
		pickle.dump(v, output_file)
		output_file.close()


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


def initNodesData(G):
	for id_node in G:
		G.node[id_node]['T_FDS'] = 0
		G.node[id_node]['C_FDS'] = 0


def norm1(est_rank, pagerank, normalization=False):
	if(normalization == False):
		return sum([abs(est_rank[n] - pagerank[n]) for n in est_rank])
	else:
		return (sum([abs((est_rank[n] - pagerank[n]) / pagerank[n]) for n in est_rank if pagerank[n] > 0]) / len(pagerank))


def normInf(est_rank, pagerank, normalization=False):
	if(normalization == False):
		return max([abs(est_rank[n] - pagerank[n]) for n in est_rank])
	else:
		return max([abs((est_rank[n] - pagerank[n]) / pagerank[n]) for n in est_rank if pagerank[n] > 0])


def fdsample(G, damping_factor=0.85, r=3):
	est_rank = {}
	for id_node in G:
		successors = G.successors(id_node)
		for i in range(G.node[id_node]['T_FDS'] + r):
			G.node[id_node]['C_FDS'] += 1
			dies = rnd.random()
			if dies < damping_factor and len(successors) > 0:
				successor = rnd.choice(successors)
				G.node[successor]['T_FDS'] += 1
		G.node[id_node]['T_FDS'] = 0
		
#	print step, " ", G.nodes(data=True)
#	print step, " ", totalElements(G, 'T_FDS')
	tot_tokens = totalElements(G, 'T_FDS')
	
	# Computing the PR
	total_C = totalElements(G, 'C_FDS')
	for id_node in G:
		est_rank[id_node] = float(G.node[id_node]['C_FDS']) / float(total_C)

	return (est_rank, tot_tokens)


def exportDataPerStep(dict_data, filename):
	dict_data = collections.OrderedDict(sorted(dict_data.items(), key=lambda t: t[0], reverse=False))
	f = open(OUTCOMES_PATH + filename, 'w')
	for key, value in dict_data.items():
		f.write(str(key) + " " + str(value) + '\n')
	f.close()


def exportPrecisionKTops(list_dict_precision_k, rate, beta, algorithm):
	beta_string = ""
	if beta != -1:
		beta_string += "_beta" + str(beta)
	f_k10 = open(OUTCOMES_PATH + 'precision_k10' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	f_k20 = open(OUTCOMES_PATH + 'precision_k20' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	f_k50 = open(OUTCOMES_PATH + 'precision_k50' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	f_k100 = open(OUTCOMES_PATH + 'precision_k100' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	f_k500 = open(OUTCOMES_PATH + 'precision_k500' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	f_k1000 = open(OUTCOMES_PATH + 'precision_k1000' + '_rate' + str(rate) + '_delta' + str(DELTA) + beta_string + '_' + algorithm + '.csv', 'w')
	for index, k in enumerate(PRECISION_K):
		dict_precision_k = list_dict_precision_k[index]
		dict_precision_k = collections.OrderedDict(sorted(dict_precision_k.items(), key=lambda t: t[0], reverse=False))
		for iteration in dict_precision_k:
			if k == 10: f_k10.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
			elif k == 20: f_k20.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
			elif k == 50: f_k50.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
			elif k == 100: f_k100.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
			elif k == 500: f_k500.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
			elif k == 1000: f_k1000.write(str(iteration) + " " + str(dict_precision_k[iteration]) + '\n')
	f_k10.close()
	f_k20.close()
	f_k50.close()
	f_k100.close()
	f_k500.close()
	f_k1000.close()
			

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


def fractionChangingEdges(dict_single_graphs, iteration):
	first_aggregated_edges_list = []
	second_aggregated_edges_list = []
	
	first_set_lower_index = iteration - DELTA + 1
	first_set_higher_index = iteration
	second_set_lower_index = iteration - (DELTA * 2) + 1
	second_set_higher_index = iteration - DELTA
	
	if second_set_lower_index < 0:
		return float(0)
	
	for i in range(first_set_lower_index, first_set_higher_index + 1):
		for edge in dict_single_graphs[i].edges():
			first_aggregated_edges_list.append(edge)
	
	for i in range(second_set_lower_index, second_set_higher_index + 1):
		for edge in dict_single_graphs[i].edges():
			second_aggregated_edges_list.append(edge)
			
	if len(first_aggregated_edges_list) > 0 and len(second_aggregated_edges_list) > 0:			
		set_first_edges = set(first_aggregated_edges_list)
		set_second_edges = set(second_aggregated_edges_list)
		
#		print "CURRENT SET\n" + str(set_first_edges) + "\n\n"
#		print "PREVIOUS SET\n" + str(set_second_edges) + "\n\n"		
		
		card_first_minus_second = len(set_first_edges - set_second_edges)
		card_second_minus_first = len(set_second_edges - set_first_edges)
		card_union_edges = len(set_first_edges | set_second_edges)
		
		return float(card_first_minus_second + card_second_minus_first) / float(card_union_edges)
	else:
		return float(0)
	

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
	initNodesData(G)
	
	dict_frac_changing_edges = {}
	dict_single_graphs = {}
	
	dict_tokens_FDS = {}
	dict_norms_1_FDS_PR = {}
	dict_norms_inf_FDS_PR = {}
	dict_norms_1_FDS_PR_normalized = {}
	dict_norms_inf_FDS_PR_normalized = {}
	list_dict_precision_k_FDS = [{}, {}, {}, {}, {}, {}]
	
	for iteration, graph_file in enumerate(data_files):
	
		print "\n\nITERATION: " + str(iteration) + ", processing " + graph_file + ", DELTA=" + str(DELTA)
		buildGraph(G, graph_file)
		print "Nodes:", len(G)
		print "Edges:", len(G.edges())
		
		print "Dangling nodes (original graph):", str(countDanglingNodes(G))
		
		# Deep-copy of the graph
		dict_single_graphs[iteration] = G.to_directed();
		
		print "\nAlgorithm FDSAMPLE..."
		ts = int(time.time())
		if checkExistingData(FDSAMPLE_DATA, rate=RATE, iterations=iteration):
			print "Importing existing data..."
			(est_rank_FDS, tokens_FDS) = getExistingData(FDSAMPLE_DATA, rate=RATE, iterations=iteration)
		else:
			(est_rank_FDS, tokens_FDS) = fdsample(G, damping_factor=ALPHA, r=RATE)
			dict_data_FDS = {'est_rank_FDS' : est_rank_FDS, 'tokens_FDS' : tokens_FDS}
			print "Writing data..."
			writeData(dict_data_FDS, rate=RATE, iterations=iteration)
		tf = int(time.time())
		print "Time elapsed: %d seconds" % (tf - ts)
		
		dict_tokens_FDS[iteration] = tokens_FDS
		
		
		
		if iteration % DELTA == DELTA - 1:
			# Fraction of changing edges
			dict_frac_changing_edges[iteration] = fractionChangingEdges(dict_single_graphs, iteration)
			
			# Computing...
			delta_graphs = []
			offset = 0
			while(offset < DELTA):
				delta_graphs.append(dict_single_graphs[iteration - offset])
				offset += 1
			
			print "\nBuilding the aggregated graph on delta graphs..."
			G_aggregated = buildAggregatedGraph(delta_graphs)
			initNodesData(G_aggregated)

#			print G_aggregated.nodes(data=True)
#			print G_aggregated.edges(data=True)
			
			print "Nodes (aggregated graph):", len(G_aggregated)
			print "Edges (aggregated graph):", len(G_aggregated.edges())
			print "Avg out-degree (aggregated graph):", np.mean([G_aggregated.out_degree(n) for n in G_aggregated])
			print "Dangling nodes (aggregated graph):", str(countDanglingNodes(G_aggregated))
			
			print "\nAlgorithm PageRank..."
			ts = int(time.time())
			if checkExistingData(PAGERANK_DATA, delta=DELTA, iterations=iteration):
				print "Importing existing data..."
				dict_rank_PR = getExistingData(PAGERANK_DATA, delta=DELTA, iterations=iteration)[0]
			else:
				dict_rank_PR = nx.pagerank(G_aggregated, alpha=ALPHA, max_iter=ITERATIONS_PR, weight='weight')
				dict_data_PR = {'dict_rank_PR' : dict_rank_PR}
				print "Writing data..."
				writeData(dict_data_PR, delta=DELTA, iterations=iteration)
			tf = int(time.time())
			print "Time elapsed: %d seconds" % (tf - ts)
			
			
			print "\nComputing L1 and LInf Norms (FDSAMPLE)..."
			dict_norms_1_FDS_PR[iteration] = norm1(est_rank_FDS, dict_rank_PR)
			dict_norms_inf_FDS_PR[iteration] = normInf(est_rank_FDS, dict_rank_PR)
			dict_norms_1_FDS_PR_normalized[iteration] = norm1(est_rank_FDS, dict_rank_PR, normalization=True)
			dict_norms_inf_FDS_PR_normalized[iteration] = normInf(est_rank_FDS, dict_rank_PR, normalization=True)
			
			print "Computing the precision@k (FDSAMPLE)..."
			for index, k in enumerate(PRECISION_K):
				real_rank_k = collections.OrderedDict(sorted(dict_rank_PR.items(), key=lambda t: t[1], reverse=True)[:k])
				est_rank_k = collections.OrderedDict(sorted(est_rank_FDS.items(), key=lambda t: t[1], reverse=True)[:k])
				frac = 0
				for key_real in real_rank_k.keys():
					if key_real in est_rank_k:
						frac += 1
				frac = float(frac) / float(k)
				list_dict_precision_k_FDS[index][iteration] = frac
				
				
		# Releasing memory... deleting old unused graphs
		if iteration >= DELTA * 2:
			print "Dropping dict_single_graphs[" + str(iteration - DELTA * 2) + "]..."
			del dict_single_graphs[iteration - (DELTA * 2)]
	
	
	print "\nExporting the outcomes..."
	exportDataPerStep(dict_frac_changing_edges, 'changing_edges_delta' + str(DELTA) + '.csv')
	
	exportDataPerStep(dict_tokens_FDS, 'tokens' + '_rate' + str(RATE) + '_delta' + str(DELTA) + '_FDS.csv')	
	exportDataPerStep(dict_norms_1_FDS_PR, 'l1_norm' + '_rate' + str(RATE) + '_delta' + str(DELTA) + '_FDS.csv')
	exportDataPerStep(dict_norms_inf_FDS_PR, 'linf_norm' + '_rate' + str(RATE) + '_delta' + str(DELTA) + '_FDS.csv')
	exportDataPerStep(dict_norms_1_FDS_PR_normalized, 'l1_norm_normalized' + '_rate' + str(RATE) + '_delta' + str(DELTA) + '_FDS.csv')
	exportDataPerStep(dict_norms_inf_FDS_PR_normalized, 'linf_norm_normalized' + '_rate' + str(RATE) + '_delta' + str(DELTA) + '_FDS.csv')
	exportPrecisionKTops(list_dict_precision_k_FDS, rate=RATE, beta=-1, algorithm='FDS')
	
	
	
if __name__ == '__main__':
	main()
