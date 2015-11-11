import numpy

WORKER_COUNT_THRESHOLD = 20
BIOLOGICAL_RESULTS_FILE_FS = './biological_dataset_fs.csv'
BIOLOGICAL_RESULTS_FILE_AA = './biological_dataset_aa.csv'
CROWD_RESULTS_FILE = './crowd_dataset.csv'
NUM_IMAGES_PER_HIT = 15

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



# Helper functions to read data and create a segmentation tree
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



from tabulate import tabulate

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
	mat.append([i, T.frontier_seeking(), T.onlyRoot(), (T1.frontier_seeking()[0], T1.frontier_seeking()[1]-1)])
	# For T1, we need to remove 1 question corresponding to the dummy root node

print tabulate(mat, headers = ['Image', 'FS (Count, #Questions)', 'OnlyRoot (Count, #Questions)', 'AA (Count, #Questions)'], tablefmt='orgtbl')



