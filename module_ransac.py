import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import trimesh
import torch
import time

def o3d_geometries_from_trimesh(sphere_params,cylinder_params,cuboid_params):
    # Initialize the list of geometries
    geometries = []

    # Define transparency and color for each primitive type
    sphere_color = [1, 0, 0]  # Red
    cylinder_color = [0, 1, 0]  # Green
    cuboid_color = [0, 0, 1]  # Blue
    alpha = 0.5  # Set transparency to 50%

    # Process spheres
    for sphere in sphere_params:
        sphere_geometry = o3d.geometry.TriangleMesh.create_sphere(radius=sphere['radius'])
        sphere_geometry.translate(sphere['center'])  # Move sphere to correct position
        sphere_geometry.paint_uniform_color(sphere_color)  # Apply color
        geometries.append(sphere_geometry)

    # Process cylinders
    for cylinder in cylinder_params:
        cylinder_geometry = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder['radius'], height=cylinder['height'])
        # Apply the transformation to place and rotate the cylinder correctly
        cylinder_geometry.transform(cylinder['transform'])
        cylinder_geometry.paint_uniform_color(cylinder_color)  # Apply color
        geometries.append(cylinder_geometry)

    # Process cuboids (boxes)
    for cuboid in cuboid_params:
        extents = cuboid['extents']
        cuboid_geometry = o3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1], depth=extents[2])
        # Apply the transformation to position and rotate the box correctly
        cuboid_geometry.transform(cuboid['transform'])
        cuboid_geometry.paint_uniform_color(cuboid_color)  # Apply color
        geometries.append(cuboid_geometry)
        
    return geometries

def point_cloud_to_tensor(pcd, num_points=2048):
    # Extract point positions and normals from the point cloud
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Center the point cloud (shift the centroid to the origin)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Find the maximum distance from the origin along any axis (for scaling)
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    
    # Scale the points so that the maximum distance is 1 (proportional scaling)
    scaled_points = centered_points / max_distance

    # Concatenate scaled positions and normals (shape: [N, 6])
    pcd_data = np.hstack((scaled_points, normals))

    # Downsample or upsample to num_points (2048)
    if pcd_data.shape[0] > num_points:
        # Downsampling: Randomly sample num_points points
        indices = np.random.choice(pcd_data.shape[0], num_points, replace=False)
        sampled_data = pcd_data[indices]
    else:
        # Upsampling: Repeat points and randomly sample to num_points
        repeat_factor = int(np.ceil(num_points / pcd_data.shape[0]))
        repeated_data = np.tile(pcd_data, (repeat_factor, 1))
        sampled_data = repeated_data[:num_points]

    # Convert to tensor and reshape to torch.Size([1, 6, 2048])
    sampled_tensor = torch.tensor(sampled_data, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

    return sampled_tensor


def primitives_from_pcd(pcd, 
                        nn=16, 
                        std_multiplier=10, 
                        max_plane_idx=6, 
                        epsilon_plane=0.15, 
                        min_cluster_points_plane=5, 
                        epsilon_cluster=0.03, 
                        min_cluster_points_rest=10, 
                        debug=False):

    # Step 1: Filter the point cloud using a statistical outlier removal method
    pcd_filtered = pcd.remove_statistical_outlier(nn, std_multiplier)
    outliers = pcd.select_by_index(pcd_filtered[1], invert=True)
    outliers.paint_uniform_color([1, 0, 0])
    pcd = pcd_filtered[0]
    
    # if debug:
    #     print("After filtering:", len(np.asarray(pcd.points)), "points")
    #     o3d.visualization.draw_geometries([pcd])

    # Step 2: Compute nearest neighbor distance and estimate normals
    nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
    radius_normals = nn_distance * 4
    if debug:
        print("Mean nearest neighbor distance:", nn_distance)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), 
                         fast_normal_computation=True)

    pt_to_plane = nn_distance

    # Step 3: Segment planes from the point cloud
    planes_pcd = []
    plane_params = []
    pcd_rest = pcd

    for i in range(max_plane_idx):
        colors = plt.get_cmap("tab20")(i)
        param, inliers = pcd_rest.segment_plane(distance_threshold=pt_to_plane, ransac_n=3, num_iterations=1000)
        points = pcd_rest.select_by_index(inliers)

        labels = np.array(points.cluster_dbscan(eps=epsilon_plane, min_points=min_cluster_points_plane))
        candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
        
        if debug:
            print(f"Plane {i} inliers: {len(inliers)}")
        if len(inliers) < 5000:
            break

        best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

        pcd_rest = pcd_rest.select_by_index(inliers, invert=True) + points.select_by_index(list(np.where(labels != best_candidate)[0]))
        points = points.select_by_index(list(np.where(labels == best_candidate)[0]))

        points.paint_uniform_color(list(colors[:3]))
        planes_pcd.append(points)
        plane_params.append(param)

    # if debug:
    #     o3d.visualization.draw_geometries([planes_pcd[i] for i in range(len(planes_pcd))])
        # o3d.visualization.draw_geometries([pcd_rest])

    # Step 4: Cluster remaining points and extract geometric primitives
    labels = np.array(pcd_rest.cluster_dbscan(eps=epsilon_cluster, min_points=min_cluster_points_rest))
    max_label = labels.max()
    num_clusters = max_label + 1

    # Color the clusters
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # if debug:
    #     o3d.visualization.draw_geometries([pcd_rest])

    tri_cloud=trimesh.PointCloud(np.asarray(pcd.points))
    tri_scene = trimesh.Scene()
    tri_scene.add_geometry(tri_cloud)

    # Prepare to save geometric primitive parameters
    sphere_params = []
    cylinder_params = []
    cuboid_params = []

    # Convert clusters to tensors and extract geometric primitives
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = pcd_rest.select_by_index(cluster_indices)
        
        cloud = trimesh.PointCloud(np.asarray(cluster_pcd.points))
        mesh = cloud.convex_hull
        bounding_primitive = mesh.bounding_primitive

        if isinstance(bounding_primitive, trimesh.primitives.Sphere):
            sphere_params.append({
                'center': bounding_primitive.primitive.center,
                'radius': bounding_primitive.primitive.radius
            })
        elif isinstance(bounding_primitive, trimesh.primitives.Cylinder):
            cylinder_params.append({
                'height': bounding_primitive.primitive.height,
                'radius': bounding_primitive.primitive.radius,
                'transform': bounding_primitive.primitive.transform
            })
        elif isinstance(bounding_primitive, trimesh.primitives.Box):
            cuboid_params.append({
                'transform': bounding_primitive.primitive.transform,
                'extents': bounding_primitive.primitive.extents
            })

        if debug:
            bounding_primitive_mesh = bounding_primitive.to_mesh()
            bounding_primitive_mesh.visual.face_colors = [100, 100, 250, 100]  # RGBA with transparency
            tri_scene.add_geometry(bounding_primitive_mesh)
            # scene = trimesh.Scene([mesh, bounding_primitive_mesh])
            # scene.show()

    if debug:
        print("Sphere Parameters:", sphere_params)
        print("Cylinder Parameters:", cylinder_params)
        print("Cuboid Parameters:", cuboid_params)
        print("Plane Parameters:", plane_params)
        tri_scene.show()

    return sphere_params, cylinder_params, cuboid_params, plane_params


if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud("scene.pcd")

    print(np.array(pcd.points).shape)

    list_elapse=[]
    
    start_time = time.time()
    for i in range(10):
        start_time = time.time()

        primitives_from_pcd(pcd,debug=0)

        elapsed_time = time.time() - start_time
        list_elapse.append(elapsed_time)
        print("Time cost: ",elapsed_time)

    
    print("max",np.max(list_elapse),"min",np.min(list_elapse),"mean",np.mean(list_elapse))





