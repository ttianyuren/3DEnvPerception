## Installation
`pip3 install -r requirements.txt`

## Workflow
### Point cloud generation from simulation
`python module_pointCloud_gen.py`
- Set up a task scence in PyBullet and render the point cloud merged from 2 RGBD camera. 


### Approximate point cloud with primitives
`python module_ransac.py`
- Cluster the merged point cloud, and approximate segments with geometric primitives.

