import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

OUTCOMES_PATH = 'outcomes/'
EPSILON = 0.53

def initR(size):
	R = []
	for i in range(size):
		l = []
		for j in range(size):
			l.append(0.0)
		R.append(l)
	return R


def computeRecurrenceMatrix(M, R):
	for i in range(len(M)):
		for j in range(len(M)):
			if M[i][j] < EPSILON:
				R[i][j] = 1.0
	return R
	
	
def plotRecurrencePlot(R):
	xcoord = []
	ycoord = []
	for i in range(len(R)):
		for j in range(len(R)):
			if R[i][j] == 1:
				xcoord.append(i)
				ycoord.append(j)

	x = np.array(xcoord)
	y = np.array(ycoord)
	plt.figure(figsize=(5, 5))
	plt.xlabel('t')
	plt.ylabel('t')
	plt.plot(x, y, marker = 's', ls = 'None', ms = 3, c = 'black')
	plt.xlim(0, len(R)-1)
	plt.ylim(0, len(R)-1)
	plt.savefig(OUTCOMES_PATH + 'recurrence_plot_spearman_e' + str(EPSILON) + '.png', format='png')
	#plt.savefig(OUTCOMES_PATH + 'recurrence_plot_l1norm_e' + str(EPSILON) + '.png', format='png')
	plt.close()
	

def main():
	
	f = open(OUTCOMES_PATH + 'correlations_matrix', 'r')
	#f = open(OUTCOMES_PATH + 'l1norm_matrix', 'r')
	M = deepcopy(pickle.load(f))
	R = initR(len(M))
	R = computeRecurrenceMatrix(M, R)
	plotRecurrencePlot(R)
	

if __name__ == '__main__':
	main()
