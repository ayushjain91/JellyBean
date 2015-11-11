import numpy
import math
import networkx as nx

WORKER_COUNT_THRESHOLD = 20
BIOLOGICAL_RESULTS_FILE_FS = './biological_dataset_fs.csv'
BIOLOGICAL_RESULTS_FILE_AA = './biological_dataset_aa.csv'
CROWD_RESULTS_FILE = './crowd_dataset.csv'
NUM_IMAGES_PER_HIT = 15

NOTE = "# This implementation of Articulation Avoidance is different from the one proposed in the paper\n# It uses additional heuristics to reduce the number of questions further by ~3%\n# The data provided, however, is for bins generated without using these additional heuristics"



def articulation_avoidance(segmentation_graph, segment_counts):
	# This implementation of Articulation Avoidance is different from the one proposed in the paper
	# It uses additional heuristics to reduce the number of questions further by ~3%
	# The data provided, however, is for bins generated without using these additional heuristics


	n = segmentation_graph.number_of_nodes()
	m = segmentation_graph.number_of_edges()
	groups = [] #list of all final groups
	count_groups = []
	remaining_nodes = G.nodes()

	H = segmentation_graph.copy()
	cutpoints = list(nx.articulation_points(H))

	#picking group starting nodes in sequence for now
	while len(remaining_nodes) > 0:
		new_center = -1 #first node in new group
		new_group = []


		for i in remaining_nodes:
			if (i not in cutpoints):
				new_center = i
				remaining_nodes.remove(i)
				H.remove_node(i)
				cutpoints = list(nx.articulation_points(H))
				break

		    
	    #if no remaining non-cutpoints
		if new_center == -1: 
		    new_center = remaining_nodes.pop()
		    H.remove_node(new_center)
		    cutpoints = list(nx.articulation_points(H))

	    #form new group
		new_group.append(new_center)
		current_size = segment_counts[new_center]

		regular_neighbors = [] # non Articulation Points
		cutpoint_neighbors = [] # Articulation Points

		for node in segmentation_graph.neighbors(new_center):
		    if node in remaining_nodes:
		        if node in cutpoints:
		            cutpoint_neighbors.append(node)
		        else:
		            regular_neighbors.append(node)
		max_segment_size = int(math.floor(WORKER_COUNT_THRESHOLD*1.0/max(segment_counts.values())))*max(segment_counts.values())
		
		while current_size < max_segment_size and len(regular_neighbors) + len(cutpoint_neighbors) > 0:
			if regular_neighbors:
			    add_next = regular_neighbors.pop()
			else:
			    add_next = cutpoint_neighbors.pop()

			if (add_next in remaining_nodes) and (segment_counts[add_next] + current_size <= max_segment_size):
				H.remove_node(add_next)
				cutpoints = list(nx.articulation_points(H))

				
	        	# Recompute cutpoint_neighbors and regular_neighbors
				for vertex in cutpoint_neighbors:
					if vertex not in cutpoints:
						cutpoint_neighbors.remove(vertex)
						regular_neighbors.append(vertex)

				for vertex in regular_neighbors:
					if vertex in cutpoints:
						cutpoint_neighbors.append(vertex)
						regular_neighbors.remove(vertex)
				
				

				new_group.append(add_next)
				current_size += segment_counts[add_next]
				remaining_nodes.remove(add_next)
				for node in segmentation_graph.neighbors(add_next):
				    if node in remaining_nodes:
				        if node in cutpoints:
				            cutpoint_neighbors.append(node)
				        else:
				            regular_neighbors.append(node)

		groups.append(new_group)
		count_groups.append(current_size)

	return groups, count_groups





class SegmentationTree():

	def __init__(self, image, workerCounts, aggFunc = numpy.median):
		self.children = []
		self.workerCounts = workerCounts
		self.img = image
		self.agg_func = aggFunc

	def add_child(self, node):
		self.children.append(node)

	def frontier_seeking(self):
		if self.agg_func(self.workerCounts) < WORKER_COUNT_THRESHOLD or len(self.children) <= 1:
			return self.agg_func(self.workerCounts), 1

		wCount = 0
		numQ = 1
		for child in self.children:
			wCount_child, numQ_child = child.frontier_seeking()
			wCount += wCount_child
			numQ += numQ_child

		return wCount, numQ

	def onlyRoot(self):
		return self.agg_func(self.workerCounts),1



# Helper functions to read data and create a segmentation tree/graph
# You may have to write your own function depending on fanout of segmentation tree and format of your data
def constructSegmentationTree_FS(img_no, database='CROWD'):
	
	if database == 'CROWD':
		results_file = CROWD_RESULTS_FILE
	else:
		results_file = BIOLOGICAL_RESULTS_FILE_FS

	import csv
	import re
	wCounts = {}


	with open(results_file, 'rb') as f:
		reader = csv.DictReader(f)

		for row in reader:
			for i in range(1, NUM_IMAGES_PER_HIT + 1):
			    url_parts = row['Input.img'+str(i)].split('/')
			    if 'IMG_' + str(img_no).zfill(2) not in url_parts[-1]:
			    	continue
			    img_name = url_parts[-1].split('.')[0]
			    segment_num = int(img_name.split('_')[-1])
			    if segment_num in wCounts:
			    	wCounts[segment_num].append(int(row['Answer.count'+str(i)]))
			    else:
			    	wCounts[segment_num] = [int(row['Answer.count'+str(i)])]
		k = wCounts.keys()
		k.sort()
		segmentationTreeNodes = {}
		for segment_num in k:
			new_node = SegmentationTree(img_no, wCounts[segment_num])
			segmentationTreeNodes[segment_num] = new_node
			if segment_num/2 >0:
				segmentationTreeNodes[segment_num/2].add_child(new_node)
	return segmentationTreeNodes[1]


def constructSegmentationTree_AA(img_no):
	
	results_file = BIOLOGICAL_RESULTS_FILE_AA

	import csv
	import re
	wCounts = {}


	with open(results_file, 'rb') as f:
		reader = csv.DictReader(f)

		for row in reader:
			for i in range(1, NUM_IMAGES_PER_HIT + 1):
			    url_parts = row['Input.img'+str(i)].split('/')
			    if 'IMG_' + str(img_no).zfill(2) not in url_parts[-1]:
			    	continue
			    img_name = url_parts[-1].split('.')[0]
			    segment_num = int(img_name.split('_')[-1])
			    if segment_num in wCounts:
			    	wCounts[segment_num].append(int(row['Answer.count'+str(i)]))
			    else:
			    	wCounts[segment_num] = [int(row['Answer.count'+str(i)])]
		k = wCounts.keys()
		k.sort()
		segmentationTreeNodes = {}
		segmentationTreeNodes[1] = SegmentationTree(img_no, [WORKER_COUNT_THRESHOLD]*5) # Dummy root node
		for segment_num in k:
			new_node = SegmentationTree(img_no, wCounts[segment_num])
			segmentationTreeNodes[segment_num + 1] = new_node
			segmentationTreeNodes[1].add_child(new_node)
	return segmentationTreeNodes[1]

def read_segmentation_graph(img_no):
	G = nx.Graph()
	segment_counts = {}
	total_count = 0
	max_segment_count = 0

	segGraphFile = './biological_dataset/IMG_'+str(img_no).zfill(2)+'/AA/segmentationGraph.txt'
	raw_graph = open(segGraphFile,'r')
	header = map(int,raw_graph.readline().split(" "))

	numV = header[0]
	numE = header[1]
	fmt = header[2]
	ncon = header[3]

	assert (fmt==10 and ncon==1), "File is not in the correct format"
	

	data = []
	for line in raw_graph:
	    data.append(map(int,line.split(" ")))

	for u in range(numV):
	    G.add_node(u+1)
	    segment_counts[u+1] = data[u][0]
	    total_count += segment_counts[u+1]
	    if max_segment_count < segment_counts[u+1]:
	        max_segment_count = segment_counts[u+1]
	    for v in range(1,len(data[u])):
	        G.add_edge(u+1,data[u][v])

	return G, segment_counts






from tabulate import tabulate

print "===========================================\nArticulation Avoidance (BIOLOGICAL DATASET)\n==========================================="
print NOTE
mat = []
for i in range(1,21):
	G, count = read_segmentation_graph(i)
	mat.append([i, len(articulation_avoidance(G, count)[0]), sum(articulation_avoidance(G, count)[1])])
print tabulate(mat, headers=['Image', 'Number of Partitions','Prior Counts (ML)'], tablefmt='orgtbl')


print "====================\nCROWD DATASET\n===================="
mat = []
for i in range(1,13):
	T = constructSegmentationTree_FS(i,'CROWD')
	mat.append([i, T.frontier_seeking(), T.onlyRoot()])
print tabulate(mat, headers = ['Image', 'FS (Count, #Questions)', 'OnlyRoot (Count, #Questions)'], tablefmt='orgtbl')


print "====================\nBIOLOGICAL DATASET\n===================="
mat = []
for i in range(1,21):
	T = constructSegmentationTree_FS(i,'BIO')
	T1 = constructSegmentationTree_AA(i)
	mat.append([i, T.frontier_seeking(), T.onlyRoot(), (T1.frontier_seeking()[0], len(T1.children))])
	# For T1, we need to remove 1 question corresponding to the dummy root node

print tabulate(mat, headers = ['Image', 'FS (Count, #Questions)', 'OnlyRoot (Count, #Questions)', 'AA (Count, #Questions)'], tablefmt='orgtbl')



