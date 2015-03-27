import glob

GRAPHS_DIR = '../datasets/facebook/singles/'
OUTCOMES_PATH = 'outcomes/'

def main():
	data_files = sorted(glob.glob(GRAPHS_DIR + "*.txt"))
	if len(data_files) == 0:
		return
	
	f_num_edges = open(OUTCOMES_PATH + 'num_edges.csv', 'w')
	
	for graph_file in data_files:
		f = open(graph_file, 'r')
		lines = f.readlines()
		f.close()
		
		num_edges = 0
		
		for line in lines:
			if not(line.startswith('#')):
				(node1, node2, relationship) = line.strip().split('\t')
				if node1 != node2:
					num_edges += 1
		
		f_num_edges.write(str(num_edges) + "\n")
		f_num_edges.flush()
	
	f_num_edges.close()

if __name__ == '__main__':
	main()
