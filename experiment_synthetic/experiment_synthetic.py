import collections, sys, time, os.path, glob, pickle
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from copy import deepcopy

GRAPH_FILE = '../datasets/graph_snap.txt'
FDSAMPLE_ENABLED = True
IDEALSAMPLE_ENABLED = True
ALPHA = 0.85
RATE = 1
ITERATIONS = 100
ITERATIONS_PR = 100
PROB_LINK = [0.05, 0.1, 0.2, 0.5, 0.8]
SIMULATIONS = 5
PRECISION_K = [10, 20, 50, 100, 500, 1000]

DATA_PATH = 'data/'
OUTCOMES_PATH = 'outcomes/'
GRAPH_DATA = ['G_IS', 'G_FDS']
PAGERANK_DATA = ['dict_rank_PR']
IDEALSAMPLE_DATA = ['est_rank_IS', 'tokens_IS']
FDSAMPLE_DATA = ['est_rank_FDS', 'tokens_FDS']


def checkExistingData(list_files, prob=-1, simulation=-1, rate=-1, iterations=-1):
	data_files = glob.glob(DATA_PATH + "*")
	if len(data_files) == 0:
		return False
		
	append_string = ""
	if prob != -1:
		append_string += "_p" + str(prob)
	if simulation != -1:
		append_string += "_sim" + str(simulation)
	if rate != -1:
		append_string += "_rate" + str(rate)
	if iterations != -1:
		append_string += "_iter" + str(iterations)
		
	for filename in list_files:
		total_filename = DATA_PATH + filename + append_string
		if total_filename not in data_files:
			return False
	return True


def getExistingData(list_files, prob=-1, simulation=-1, rate=-1, iterations=-1):
	output_files = []
	
	append_string = ""
	if prob != -1:
		append_string += "_p" + str(prob)
	if simulation != -1:
		append_string += "_sim" + str(simulation)
	if rate != -1:
		append_string += "_rate" + str(rate)
	if iterations != -1:
		append_string += "_iter" + str(iterations)
	
	for filename in list_files:
		total_filename = DATA_PATH + filename + append_string
		pkl_file = open(total_filename, 'rb')
		output_files.append(deepcopy(pickle.load(pkl_file)))
		pkl_file.close()
		
	return output_files


def writeData(dict_data, prob=-1, simulation=-1, rate=-1, iterations=-1):
	append_string = ""
	if prob != -1:
		append_string += "_p" + str(prob)
	if simulation != -1:
		append_string += "_sim" + str(simulation)
	if rate != -1:
		append_string += "_rate" + str(rate)
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


def initNodesData(G1, G2):
	for id_node in G1:
		G1.node[id_node]['T'] = 0
		G1.node[id_node]['C'] = 0
		G1.node[id_node]['R'] = 0
		
		G2.node[id_node]['T'] = 0
		G2.node[id_node]['C'] = 0
		G2.node[id_node]['R'] = 0


def dictRankPRMapping(dict_rank_PR):
	dict_rank_PR_mapping = {}
	i = 1
	for k in dict_rank_PR.keys():
		dict_rank_PR_mapping[k] = i
		i += 1
	return dict_rank_PR_mapping


def keysRemapping(dict_rank_PR_mapping, dict_rank):
	dict_rank_remapped = {}
	for k, v in dict_rank_PR_mapping.items():
		dict_rank_remapped[v] = dict_rank[k]
	return dict_rank_remapped


def norm1(est_rank, pagerank, normalization=False):
	if(normalization == False):
		return sum([abs(est_rank[n] - pagerank[n]) for n in pagerank])
	else:
		return (sum([abs((est_rank[n] - pagerank[n]) / pagerank[n]) for n in pagerank if pagerank[n] > 0]) / len(pagerank))


def normInf(est_rank, pagerank, normalization=False):
	if(normalization == False):
		return max([abs(est_rank[n] - pagerank[n]) for n in pagerank])
	else:
		return max([abs((est_rank[n] - pagerank[n]) / pagerank[n]) for n in pagerank if pagerank[n] > 0])


def compNormsL1InfEA(dict_all_ranks_FDS, dict_all_ranks_IS, normalization=False):
	list_norms_1_est_real = []
	list_norms_inf_est_real = []
	for iteration in range(len(dict_all_ranks_IS)):
		list_norms_1_est_real.append(norm1(dict_all_ranks_FDS[iteration], dict_all_ranks_IS[iteration], normalization))
		list_norms_inf_est_real.append(normInf(dict_all_ranks_FDS[iteration], dict_all_ranks_IS[iteration], normalization))
	return (list_norms_1_est_real, list_norms_inf_est_real)


def idealsample(G, damping_factor=0.85, r=3):
	est_rank = {}
	for id_node in G:
		successors = G.successors(id_node)
		for i in range(G.node[id_node]['T'] + r):
			dies = rnd.random()
			if dies > damping_factor:
				G.node[id_node]['C'] += 1
			else:
				if len(successors) > 0:
					successor = rnd.choice(successors)
					G.node[successor]['T'] += 1
				else:
					node_uar = rnd.choice(G.nodes())
					G.node[node_uar]['T'] += 1
		G.node[id_node]['T'] = 0
		
#	print step, " ", G.nodes(data=True)
#	print step, " ", totalElements(G, 'T')
	tot_tokens = totalElements(G, 'T')
	
	# Computing the PR
	total_C = totalElements(G, 'C')
	for id_node in G:
		est_rank[id_node] = float(G.node[id_node]['C']) / float(total_C)
		
	return (est_rank, tot_tokens)


def fdsample(G, damping_factor=0.85, r=3):
	est_rank = {}
	for id_node in G:
		successors = G.successors(id_node)
		for i in range(G.node[id_node]['T'] + r):
			dies = rnd.random()
			if dies > damping_factor:
				G.node[id_node]['C'] += 1
			else:
				if len(successors) > 0:
					successor = rnd.choice(successors)
					G.node[successor]['T'] += 1
		G.node[id_node]['T'] = 0
		
#	print step, " ", G.nodes(data=True)
#	print step, " ", totalElements(G, 'T')
	tot_tokens = totalElements(G, 'T')
	
	# Computing the PR
	total_C = totalElements(G, 'C')
	for id_node in G:
		est_rank[id_node] = float(G.node[id_node]['C']) / float(total_C)

	return (est_rank, tot_tokens)


# pagerankImpl is the source code of nx.pagerank(...)
def pagerankImpl(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-8, nstart=None, weight='weight'):
	if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
		raise Exception("pagerank() not defined for graphs with multiedges.")

	if len(G) == 0:
		return {}

	if not G.is_directed():
		D=G.to_directed()
	else:
		D=G

	# create a copy in (right) stochastic form
	W=nx.stochastic_graph(D, weight=weight)
	scale=1.0/W.number_of_nodes()

	# choose fixed starting vector if not given
	if nstart is None:
		x=dict.fromkeys(W,scale)
	else:
		x=nstart
		# normalize starting vector to 1
		s=1.0/sum(x.values())
		for k in x: x[k]*=s

	# assign uniform personalization/teleportation vector if not given
	if personalization is None:
		p=dict.fromkeys(W,scale)
	else:
		p=personalization
		# normalize starting vector to 1
		s=1.0/sum(p.values())
		for k in p:
			p[k]*=s
		if set(p)!=set(G):
			raise Exception('Personalization vector must have a value for every node')
	# "dangling" nodes, no links out from them
	out_degree=W.out_degree()
	dangle=[n for n in W if out_degree[n]==0.0]
	i=0
	while True: # power iteration: make up to max_iter iterations
		xlast=x
		x=dict.fromkeys(xlast.keys(),0)
		danglesum=alpha*scale*sum(xlast[n] for n in dangle)
		for n in x:
			# this matrix multiply looks odd because it is
			# doing a left multiply x^T=xlast^T*W
			for nbr in W[n]:
				x[nbr]+=alpha*xlast[n]*W[n][nbr][weight]
			x[n]+=danglesum+(1.0-alpha)*p[n]
		# normalize vector
		s=1.0/sum(x.values())
		for n in x:
			x[n]*=s
		
		# check convergence, l1 norm
		err=sum([abs(x[n]-xlast[n]) for n in x])
		if err < tol:
			break
		if i>max_iter:
			raise Exception('pagerank: power iteration failed to converge in %d iterations.'%(i-1))
		i+=1
	return x


def exportDataPerStep(dict_data, filename):
	dict_data = collections.OrderedDict(sorted(dict_data.items(), key=lambda t: t[0], reverse=False))
	f = open(OUTCOMES_PATH + filename, 'w')
	for key, value in dict_data.items():
		f.write(str(key) + " " + str(value) + '\n')
	f.close()


def exportRelativeError(est_rank, real_rank, filename):
	real_rank_sorted_by_value = collections.OrderedDict(sorted(real_rank.items(), key=lambda t: t[1], reverse=True))
	rel_error = [abs(est_rank[n]-real_rank_sorted_by_value[n])/real_rank_sorted_by_value[n] for n in real_rank_sorted_by_value]
	c1 = real_rank_sorted_by_value.keys()
	c2 = rel_error
	f = open(OUTCOMES_PATH + filename, 'w')
	for i, item in enumerate(c1):
		f.write(str(item) + " " + str(c2[i]) + '\n')
	f.close()


def exportPrecisionKTops(list_dict_precision_k, prob, rate, iterations, algorithm):
	f_k10 = open(OUTCOMES_PATH + 'precision_k10_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	f_k20 = open(OUTCOMES_PATH + 'precision_k20_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	f_k50 = open(OUTCOMES_PATH + 'precision_k50_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	f_k100 = open(OUTCOMES_PATH + 'precision_k100_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	f_k500 = open(OUTCOMES_PATH + 'precision_k500_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	f_k1000 = open(OUTCOMES_PATH + 'precision_k1000_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv', 'w')
	for index, k in enumerate(PRECISION_K):
		dict_precision_k = list_dict_precision_k[index]
		for iteration in range(len(dict_precision_k.keys())):
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


def exportAllNorms(list_norms, prob, rate, iterations, algorithm):
	for i in range(len(list_norms)):
		c1 = np.arange(len(list_norms[i]))
		c2 = list_norms[i]
		filename = ""
		if(i == 0): filename += 'EA_l1_norm_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv'
		elif(i == 1): filename += 'EA_linf_norm_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv'
		elif(i == 2): filename += 'EA_l1_norm_normalized_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv'
		elif(i == 3): filename += 'EA_linf_norm_normalized_p' + str(prob) + '_rate' + str(rate) + '_iter' + str(iterations) + '_' + algorithm + '.csv'
		f = open(OUTCOMES_PATH + filename, 'w')
		for j, item in enumerate(c1):
			f.write(str(item) + " " + str(c2[j]) + '\n')
		f.close()
			

def buildStaticGraph():
	f = open(GRAPH_FILE, 'r')
	lines = f.readlines()
#	print len(lines)
	f.close()
	
	edges = []
	for line in lines:
		(src, trg) = line.split()
		if src != trg:
			edge = [src, trg]
			edges.append(edge)
		
	return nx.DiGraph(edges)


def buildEvolvingGraphs(G_complete, G_IS, G_FDS, p):
	for edge in G_complete.edges():
		prob_edge = rnd.random()
		if p <= prob_edge:
			if G_IS.has_edge(*edge) and G_FDS.has_edge(*edge):
				G_IS.remove_edge(*edge)
				G_FDS.remove_edge(*edge)
		else:
			if not G_IS.has_edge(*edge) and not G_FDS.has_edge(*edge):
				G_IS.add_edge(*edge)
				G_FDS.add_edge(*edge)
	

def main():
	print "Building the graph..."
	G_complete = buildStaticGraph()
	
	for p in PROB_LINK:
		dict_dangling_nodes = {}
		
		dict_avg_PR_single_iter = {}
		
		dict_tokens_IS = {}
		dict_all_ranks_IS = {}
		dict_norms_1_IS_PR = {}
		dict_norms_inf_IS_PR = {}
		dict_norms_1_IS_PR_normalized = {}
		dict_norms_inf_IS_PR_normalized = {}
		dict_avg_rank_IS_single_iter = {}
		list_dict_precision_k_IS = [{}, {}, {}, {}, {}, {}]
		
		dict_tokens_FDS = {}
		dict_all_ranks_FDS = {}
		dict_norms_1_FDS_PR = {}
		dict_norms_inf_FDS_PR = {}
		dict_norms_1_FDS_PR_normalized = {}
		dict_norms_inf_FDS_PR_normalized = {}
		dict_avg_rank_FDS_single_iter = {}
		list_dict_precision_k_FDS = [{}, {}, {}, {}, {}, {}]
		
		for sim in range(SIMULATIONS):
			print "\n\nCloning the graph for IDEALSAMPLE and FDSAMPLE..."
			G_IS = G_complete.to_directed()
			G_FDS = G_complete.to_directed()
			initNodesData(G_IS, G_FDS)
			for iteration in range(ITERATIONS):
				print "\n\nITERATION: " + str(iteration) + ", SIMULATION: " + str(sim) + ", PROBABILITY: " + str(p)
				
				if checkExistingData(GRAPH_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration):
					print "Importing existing graph..."
					(G_IS, G_FDS) = getExistingData(GRAPH_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration)
				else:
					buildEvolvingGraphs(G_complete, G_IS, G_FDS, p)
					dict_data_G = {'G_IS' : G_IS, 'G_FDS' : G_FDS}
					print "Writing the graph..."
					writeData(dict_data_G, prob=p, simulation=sim, rate=RATE, iterations=iteration)
				
				print "\nStatistics..."
				print "Nodes:", len(G_IS.nodes())
				print "Avg out-degree:", np.mean([G_IS.out_degree(n) for n in G_IS])
				
				if iteration in dict_dangling_nodes:
					dict_dangling_nodes[iteration] += countDanglingNodes(G_IS)
				else:
					dict_dangling_nodes[iteration] = countDanglingNodes(G_IS)
				print "Dangling nodes:", str(float(dict_dangling_nodes[iteration])/float(sim+1))
				
				
				print "\nAlgorithm PageRank..."
				ts = int(time.time())
				if checkExistingData(PAGERANK_DATA, prob=p, simulation=sim, iterations=iteration):
					print "Importing existing data..."
					dict_rank_PR = getExistingData(PAGERANK_DATA, prob=p, simulation=sim, iterations=iteration)[0] # rename dict_rank_PR to dict_rank_PR1 if istruction_restricted is enabled.
				else:
					dict_rank_PR = pagerankImpl(G_IS, alpha=ALPHA, max_iter=ITERATIONS_PR)
					dict_data_PR = {'dict_rank_PR' : dict_rank_PR}
					print "Writing data..."
					writeData(dict_data_PR, prob=p, simulation=sim, iterations=iteration)
#				dict_rank_PR = collections.OrderedDict(sorted(dict_rank_PR1.items(), key=lambda t: t[1], reverse=True)[:1000]) #istruction_restricted
				
				if iteration == ITERATIONS - 1:
					print "\nSaving the rank vector for the " + str(iteration) + "-th iteration..."
					for n in dict_rank_PR:
						if n in dict_avg_PR_single_iter:
							dict_avg_PR_single_iter[n] += dict_rank_PR[n]
						else:
							dict_avg_PR_single_iter[n] = dict_rank_PR[n]
				
				tf = int(time.time())
				print "Time elapsed: %d seconds" % (tf - ts)
	
	
				if IDEALSAMPLE_ENABLED:
					print "\nAlgorithm IDEALSAMPLE..."
					ts = int(time.time())
					if checkExistingData(IDEALSAMPLE_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration):
						print "Importing existing data..."
						(est_rank_IS, tokens_IS) = getExistingData(IDEALSAMPLE_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration)
					else:
						(est_rank_IS, tokens_IS) = idealsample(G_IS, damping_factor=ALPHA, r=RATE)
						dict_data_IS = {'est_rank_IS' : est_rank_IS, 'tokens_IS' : tokens_IS}
						print "Writing data..."
						writeData(dict_data_IS, prob=p, simulation=sim, rate=RATE, iterations=iteration)
					tf = int(time.time())
					print "Time elapsed: %d seconds" % (tf - ts)
					
					print "\nSaving the rank..."
					if iteration in dict_all_ranks_IS:
						est_rank_per_iteration = dict_all_ranks_IS[iteration]
						for id_node in est_rank_per_iteration:
							est_rank_per_iteration[id_node] += est_rank_IS[id_node]
					else:
						dict_all_ranks_IS[iteration] = est_rank_IS
					
					print "\nSaving tokens..."
					if iteration in dict_tokens_IS:
						dict_tokens_IS[iteration] += tokens_IS
					else:
						dict_tokens_IS[iteration] = tokens_IS
					
					print "\nComputing L1 and LInf Norms..."
					if iteration in dict_norms_1_IS_PR:
						dict_norms_1_IS_PR[iteration] += norm1(est_rank_IS, dict_rank_PR)
						dict_norms_inf_IS_PR[iteration] += normInf(est_rank_IS, dict_rank_PR)
						dict_norms_1_IS_PR_normalized[iteration] += norm1(est_rank_IS, dict_rank_PR, normalization=True)
						dict_norms_inf_IS_PR_normalized[iteration] += normInf(est_rank_IS, dict_rank_PR, normalization=True)
					else:
						dict_norms_1_IS_PR[iteration] = norm1(est_rank_IS, dict_rank_PR)
						dict_norms_inf_IS_PR[iteration] = normInf(est_rank_IS, dict_rank_PR)
						dict_norms_1_IS_PR_normalized[iteration] = norm1(est_rank_IS, dict_rank_PR, normalization=True)
						dict_norms_inf_IS_PR_normalized[iteration] = normInf(est_rank_IS, dict_rank_PR, normalization=True)
		
					if iteration == ITERATIONS - 1:
						print "\nSaving the rank vector for the " + str(iteration) + "-th iteration..."
						for n in est_rank_IS:
							if n in dict_avg_rank_IS_single_iter:
								dict_avg_rank_IS_single_iter[n] += est_rank_IS[n]
							else:
								dict_avg_rank_IS_single_iter[n] = est_rank_IS[n]
					
					print "\nComputing the precision@k..."
					for index, k in enumerate(PRECISION_K):
						real_rank_k = collections.OrderedDict(sorted(dict_rank_PR.items(), key=lambda t: t[1], reverse=True)[:k])
						est_rank_k = collections.OrderedDict(sorted(est_rank_IS.items(), key=lambda t: t[1], reverse=True)[:k])
						frac = 0
						for key_real in real_rank_k.keys():
							if key_real in est_rank_k:
								frac += 1
						frac = float(frac) / float(k)
						
						dict_precision_k = list_dict_precision_k_IS[index]
						if iteration in dict_precision_k:
							dict_precision_k[iteration] += frac
						else:
							dict_precision_k[iteration] = frac
		
		
				if FDSAMPLE_ENABLED:
					print "\nAlgorithm FDSAMPLE..."
					ts = int(time.time())
					if checkExistingData(FDSAMPLE_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration):
						print "Importing existing data..."
						(est_rank_FDS, tokens_FDS) = getExistingData(FDSAMPLE_DATA, prob=p, simulation=sim, rate=RATE, iterations=iteration)
					else:
						(est_rank_FDS, tokens_FDS) = fdsample(G_FDS, damping_factor=ALPHA, r=RATE)
						dict_data_FDS = {'est_rank_FDS' : est_rank_FDS, 'tokens_FDS' : tokens_FDS}
						print "Writing data..."
						writeData(dict_data_FDS, prob=p, simulation=sim, rate=RATE, iterations=iteration)
					tf = int(time.time())
					print "Time elapsed: %d seconds" % (tf - ts)
					
					print "\nSaving the rank..."
					if iteration in dict_all_ranks_FDS:
						est_rank_per_iteration = dict_all_ranks_FDS[iteration]
						for id_node in est_rank_per_iteration:
							est_rank_per_iteration[id_node] += est_rank_FDS[id_node]
					else:
						dict_all_ranks_FDS[iteration] = est_rank_FDS
					
					print "\nSaving tokens..."
					if iteration in dict_tokens_FDS:
						dict_tokens_FDS[iteration] += tokens_FDS
					else:
						dict_tokens_FDS[iteration] = tokens_FDS
					
					print "\nComputing L1 and LInf Norms..."
					if iteration in dict_norms_1_FDS_PR:
						dict_norms_1_FDS_PR[iteration] += norm1(est_rank_FDS, dict_rank_PR)
						dict_norms_inf_FDS_PR[iteration] += normInf(est_rank_FDS, dict_rank_PR)
						dict_norms_1_FDS_PR_normalized[iteration] += norm1(est_rank_FDS, dict_rank_PR, normalization=True)
						dict_norms_inf_FDS_PR_normalized[iteration] += normInf(est_rank_FDS, dict_rank_PR, normalization=True)
					else:
						dict_norms_1_FDS_PR[iteration] = norm1(est_rank_FDS, dict_rank_PR)
						dict_norms_inf_FDS_PR[iteration] = normInf(est_rank_FDS, dict_rank_PR)
						dict_norms_1_FDS_PR_normalized[iteration] = norm1(est_rank_FDS, dict_rank_PR, normalization=True)
						dict_norms_inf_FDS_PR_normalized[iteration] = normInf(est_rank_FDS, dict_rank_PR, normalization=True)

					if iteration == ITERATIONS - 1:
						print "\nSaving the rank vector for the " + str(iteration) + "-th iteration..."
						for n in est_rank_FDS:
							if n in dict_avg_rank_FDS_single_iter:
								dict_avg_rank_FDS_single_iter[n] += est_rank_FDS[n]
							else:
								dict_avg_rank_FDS_single_iter[n] = est_rank_FDS[n]
					
					print "\nComputing the precision@k..."
					for index, k in enumerate(PRECISION_K):
						real_rank_k = collections.OrderedDict(sorted(dict_rank_PR.items(), key=lambda t: t[1], reverse=True)[:k])
						est_rank_k = collections.OrderedDict(sorted(est_rank_FDS.items(), key=lambda t: t[1], reverse=True)[:k])
						frac = 0
						for key_real in real_rank_k.keys():
							if key_real in est_rank_k:
								frac += 1
						frac = float(frac) / float(k)
						
						dict_precision_k = list_dict_precision_k_FDS[index]
						if iteration in dict_precision_k:
							dict_precision_k[iteration] += frac
						else:
							dict_precision_k[iteration] = frac
			
		# Normalization for simulation
		print "\nNormalizing over simulations..."		
		for n in range(ITERATIONS):
			dict_dangling_nodes[n] = float(dict_dangling_nodes[n]) / float(SIMULATIONS)
			
			for id_node in dict_all_ranks_IS[iteration]:
				dict_all_ranks_IS[iteration][id_node] /= float(SIMULATIONS)
			dict_tokens_IS[n] = float(dict_tokens_IS[n]) / float(SIMULATIONS)
			dict_norms_1_IS_PR[n] = float(dict_norms_1_IS_PR[n]) / float(SIMULATIONS)
			dict_norms_inf_IS_PR[n] = float(dict_norms_inf_IS_PR[n]) / float(SIMULATIONS)
			dict_norms_1_IS_PR_normalized[n] = float(dict_norms_1_IS_PR_normalized[n]) / float(SIMULATIONS)
			dict_norms_inf_IS_PR_normalized[n] = float(dict_norms_inf_IS_PR_normalized[n]) / float(SIMULATIONS)
			
			for id_node in dict_all_ranks_FDS[iteration]:
				dict_all_ranks_FDS[iteration][id_node] /= float(SIMULATIONS)
			dict_tokens_FDS[n] = float(dict_tokens_FDS[n]) / float(SIMULATIONS)
			dict_norms_1_FDS_PR[n] = float(dict_norms_1_FDS_PR[n]) / float(SIMULATIONS)
			dict_norms_inf_FDS_PR[n] = float(dict_norms_inf_FDS_PR[n]) / float(SIMULATIONS)
			dict_norms_1_FDS_PR_normalized[n] = float(dict_norms_1_FDS_PR_normalized[n]) / float(SIMULATIONS)
			dict_norms_inf_FDS_PR_normalized[n] = float(dict_norms_inf_FDS_PR_normalized[n]) / float(SIMULATIONS)
			
		for n in dict_rank_PR:
			dict_avg_PR_single_iter[n] = float(dict_avg_PR_single_iter[n]) / float(SIMULATIONS)
			dict_avg_rank_IS_single_iter[n] = float(dict_avg_rank_IS_single_iter[n]) / float(SIMULATIONS)
			dict_avg_rank_FDS_single_iter[n] = float(dict_avg_rank_FDS_single_iter[n]) / float(SIMULATIONS)
			
		for dict_precision_k in list_dict_precision_k_IS:
			for n in range(ITERATIONS):
				dict_precision_k[n] = float(dict_precision_k[n]) / float(SIMULATIONS)
				
		for dict_precision_k in list_dict_precision_k_FDS:
			for n in range(ITERATIONS):
				dict_precision_k[n] = float(dict_precision_k[n]) / float(SIMULATIONS)
		
		print "\nRemapping dictionary keys..."
		dict_avg_PR_single_iter_mapping = dictRankPRMapping(dict_avg_PR_single_iter)
		dict_avg_PR_single_iter_remapped = keysRemapping(dict_avg_PR_single_iter_mapping, dict_avg_PR_single_iter)
		dict_avg_rank_IS_single_iter_remapped = keysRemapping(dict_avg_PR_single_iter_mapping, dict_avg_rank_IS_single_iter)
		dict_avg_rank_FDS_single_iter_remapped = keysRemapping(dict_avg_PR_single_iter_mapping, dict_avg_rank_FDS_single_iter)
		
		print "\nExporting the outcomes..."
		exportDataPerStep(dict_dangling_nodes, 'dangling_nodes_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS))
		
		exportDataPerStep(dict_tokens_IS, 'tokens_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportDataPerStep(dict_norms_1_IS_PR, 'l1_norm_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportDataPerStep(dict_norms_inf_IS_PR, 'linf_norm_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportDataPerStep(dict_norms_1_IS_PR_normalized, 'l1_norm_normalized_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportDataPerStep(dict_norms_inf_IS_PR_normalized, 'linf_norm_normalized_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportRelativeError(dict_avg_rank_IS_single_iter_remapped, dict_avg_PR_single_iter_remapped, 'relative_error_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_IS.csv')
		exportPrecisionKTops(list_dict_precision_k_IS, prob=p, rate=RATE, iterations=ITERATIONS, algorithm='IS')
		
		exportDataPerStep(dict_tokens_FDS, 'tokens_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportDataPerStep(dict_norms_1_FDS_PR, 'l1_norm_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportDataPerStep(dict_norms_inf_FDS_PR, 'linf_norm_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportDataPerStep(dict_norms_1_FDS_PR_normalized, 'l1_norm_normalized_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportDataPerStep(dict_norms_inf_FDS_PR_normalized, 'linf_norm_normalized_rate_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportRelativeError(dict_avg_rank_FDS_single_iter_remapped, dict_avg_PR_single_iter_remapped, 'relative_error_p' + str(p) + '_rate' + str(RATE) + '_iter' + str(ITERATIONS) + '_FDS.csv')
		exportPrecisionKTops(list_dict_precision_k_FDS, prob=p, rate=RATE, iterations=ITERATIONS, algorithm='FDS')
		
		# E[A] using IDEALSAMPLE instead of the real E[A] on true Pagerank
		(list_norms_1_IS_FDS, list_norms_inf_IS_FDS) = compNormsL1InfEA(dict_all_ranks_FDS, dict_all_ranks_IS, normalization=False)
		(list_norms_1_IS_FDS_normalization, list_norms_inf_IS_FDS_normalization) = compNormsL1InfEA(dict_all_ranks_FDS, dict_all_ranks_IS, normalization=True)
		exportAllNorms([list_norms_1_IS_FDS, list_norms_inf_IS_FDS, list_norms_1_IS_FDS_normalization, list_norms_inf_IS_FDS_normalization], prob=p, rate=RATE, iterations=ITERATIONS, algorithm='IS_FDS')

if __name__ == '__main__':
	main()
