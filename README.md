# JellyBean

This repository contains code and data for the paper
[*Surpassing Humans and Computers with JELLYBEAN: Crowd-Vision-Hybrid Counting Algorithms*]
(http://www.aaai.org/ocs/index.php/HCOMP/HCOMP15/paper/viewFile/11593/11440).

### Repository Structure
- *JellyBean.py*: Python file containing codes for FrontierSeeking and Articulation Avoidance Algorithms
- *worker_behavior_dataset.csv*: CSV File containing worker responses for preliminary experiments (Data used for Figure 2)
- *crowd_dataset.csv*: CSV File containing worker responses for FrontierSeeking on crowd dataset
- *biological_dataset_fs.py*: CSV File containing worker responses for FrontierSeeking on biological dataset
- *biological_dataset_aa.py*: CSV File containing worker responses for ArticulationAvoidance on biological dataset
- *biological_dataset*: Directory containing images for the biological dataset
  - *IMG_01*: Directory containing segments of image IMG_01
    - *count.txt*: File containing the ground truth IMG_01
    - *FS*: Directory containing the segments for FrontierSeeking algorithm. IMG_01_01 is the root node of the segmentation tree. IMG_01_{i}'s parent in the segmentation tree is IMG_01_{i/2}
    - *AA*: Directory containing the segments for ArticulationAvoidance algorithm, IMG_01_{$i} is the i-th partition
    - *IMG_01cells.png*: Actual image
- *crowd_dataset*: Directory containing images for the crowd dataset
  - *IMG_01.jpg*: Image in the dataset
  - *IMG_01*: Directory containing segments for IMG_01.jpg. IMG_01_01 is the root node of the segmentation tree. IMG_01_{i}'s parent in the segmentation tree is IMG_01_{i/2}
    - *groundTruth.txt*: File containing the ground truth IMG_01

### Running the Code
```
python JellyBean.py
```

**NOTE:** The implementation of Articulation Avoidance here is different from the one proposed in the paper. It uses additional heuristics to reduce the number of questions further. The data provided, however, is for bins generated without using these additional heuristics

    
