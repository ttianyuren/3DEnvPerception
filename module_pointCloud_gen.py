import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
from PIL import Image


def save_depth_image(depth_array, near, far, file_name):
    # Normalize depth values to the range [0, 255]
    normalized_depth = (depth_array - near) / (far - near) * 255
    normalized_depth = np.clip(normalized_depth, 0, 255)  # Ensure values are within [0, 255]

    # Convert to an unsigned 8-bit integer array (grayscale image format)
    depth_image = normalized_depth.astype(np.uint8)

    # Create a PIL image from the numpy array
    img = Image.fromarray(depth_image)

    # Save the image as BMP
    img.save(file_name, format='BMP')


def visualize_robot_mask(robot_mask):
    """
    Visualizes the robot mask using matplotlib.

    Parameters:
        robot_mask (numpy.ndarray): The binary mask of the robot (2D array).
    """
    plt.figure(figsize=(6, 6))
    # Display the robot mask using a grayscale colormap
    plt.imshow(robot_mask, cmap="gray")
    plt.title("Robot Mask Visualization")
    plt.colorbar(
        label="Mask Value"
    )  # Adds a color bar to indicate 0 (background) and 1 (robot)
    plt.show()


def visualize_point_cloud(point_cloud):
    # Visualize point cloud
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of point cloud
    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        s=1,
        c=point_cloud[:, 2],
        cmap="jet",
    )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axes limits to make sure they have the same scale
    max_range = (
        np.array(
            [
                point_cloud[:, 0].max() - point_cloud[:, 0].min(),
                point_cloud[:, 1].max() - point_cloud[:, 1].min(),
                point_cloud[:, 2].max() - point_cloud[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
    mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
    mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set title and show plot
    plt.title("Point Cloud Visualization")
    plt.show()


def voxel_downsample(point_cloud: np.ndarray, voxel_size: float):
    """
    Downsamples a point cloud using voxel grid filtering.

    Args:
        point_cloud (np.ndarray): Input point cloud as a Nx3 numpy array.
        voxel_size (float): Size of the voxel grid.

    Returns:
        np.ndarray: Downsampled point cloud as a Mx3 numpy array.
    """
    # Convert numpy array to open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Apply voxel grid downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # # Convert back to numpy array
    # downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_pcd


def save_point_cloud_as_obj_with_trimesh(point_cloud, file_path):
    """
    Save a point cloud as an .obj file using trimesh.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3), where N is the number of points,
                   and each row contains the x, y, z coordinates of a point.
    - file_path: The path where the .obj file will be saved.
    """
    # Create a trimesh PointCloud object
    cloud = trimesh.PointCloud(point_cloud)

    # Export the point cloud to the .obj format and write to file
    with open(file_path, "w") as file:
        file.write(cloud.export(file_type="obj"))


def draw_frame(tform, life_time=1000, high_light=False):
    length = 0.7
    width = 3

    po = tform[:3, 3]
    px = tform[:3, 0] * length + po
    py = tform[:3, 1] * length + po
    pz = tform[:3, 2] * length + po

    cx = (1, 0, 0)
    cy = (0, 1, 0)
    cz = (0, 0, 1)

    if high_light:
        cx = (1, 0.7, 0.7)
        cy = (0.7, 1, 0.7)
        cz = (0.7, 0.7, 1)

    line_x = p.addUserDebugLine(po, px, cx, width, lifeTime=life_time)
    line_y = p.addUserDebugLine(po, py, cy, width, lifeTime=life_time)
    line_z = p.addUserDebugLine(po, pz, cz, width, lifeTime=life_time)

    return [line_x, line_y, line_z]


def calculate_camera_rotation_matrix(
    camera_position, target_position, up_world=np.array([0, 0, 1])
):
    # Step 1: Calculate the forward vector (z-axis)
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)  # Normalize the forward vector

    # Step 2: Calculate the right vector (x-axis)
    right = np.cross(up_world, forward)
    right = right / np.linalg.norm(right)  # Normalize the right vector

    # Step 3: Calculate the up vector (y-axis)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)  # Normalize the up vector

    # Step 4: Construct the rotation matrix
    rotation_matrix = np.array(
        [right, up, -forward]
    ).T  # Transpose to match the correct form

    return rotation_matrix


def calculate_camera_pose(
    camera_position, target_position, up_world=np.array([0, 0, 1])
):
    # Step 1: Calculate the forward vector (z-axis)
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)  # Normalize the forward vector

    # Step 2: Calculate the right vector (x-axis)
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)  # Normalize the right vector

    # Step 3: Calculate the down vector (y-axis)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)  # Normalize the up vector

    # Step 4: Construct the 3x3 rotation matrix
    rotation_matrix = np.array(
        [right, down, forward]
    ).T  # Transpose to match the correct form

    # Step 5: Construct the full 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Top-left 3x3 for rotation
    transformation_matrix[:3, 3] = (
        camera_position  # Set the translation (camera position)
    )

    return transformation_matrix


def random_pose(x=[-0.8,0.8],y=[-0.8,0.8],z=[0.,1.8]):
    position = [random.uniform(x[0], x[1]), random.uniform(y[0], y[1]), random.uniform(z[0], z[1])]
    orientation = p.getQuaternionFromEuler(
        [
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        ]
    )
    return position, orientation


def random_size(min_size, max_size):
    return random.uniform(min_size, max_size)


def apply_damping(object_id, linear_damping=0.9, angular_damping=0.9):
    p.changeDynamics(
        object_id, -1, linearDamping=linear_damping, angularDamping=angular_damping
    )


def create_random_sphere(min_size=0.05, max_size=0.5):
    radius = random_size(min_size, max_size)
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]
    )
    sphere_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(sphere_id)


def create_random_cuboid(min_size=0.05, max_size=0.5):
    half_extents = [
        random_size(min_size, max_size) / 2,
        random_size(min_size, max_size) / 2,
        random_size(min_size, max_size) / 2,
    ]
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0, 1, 0, 1]
    )
    cuboid_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(cuboid_id)


def create_random_cylinder(min_size=0.05, max_size=0.5):
    radius = random_size(min_size, max_size) / 2
    height = random_size(min_size * 2, max_size * 3)
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0, 0, 1, 1]
    )
    cylinder_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(cylinder_id)


def capture_rgbd_image(
    camera_position,
    target_pos,
    width=640,
    height=480,
    fov_v=57,
    near=0.01,
    far=10,
    robot_id=None,
):
    aspect = width / height

    # Compute the view and projection matrices
    view_matrix = p.computeViewMatrix(camera_position, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov_v, aspect, near, far)

    # Get the camera image (RGBA, Depth, Segmentation)
    image = p.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,  # To get the segmentation mask
    )

    # Extract the RGB image from the result
    rgb_image = np.reshape(image[2], (height, width, 4))[
        :, :, :3
    ]  # Extract RGB (ignore Alpha channel)

    # Extract the depth buffer
    depth_buffer = np.reshape(image[3], (height, width))

    # Convert the depth buffer to actual depth values
    depth_image = far * near / (far - (far - near) * depth_buffer)

    robot_mask = None
    if robot_id is not None:
        # Extract the segmentation mask
        segmentation_mask = np.reshape(image[4], (height, width))

        # The object ID is stored in the upper 24 bits of the segmentation mask
        # We mask out the lower 8 bits to compare the object ID part only
        object_id_mask = segmentation_mask & ((1 << 24) - 1)  # Extract object ID

        # Create a binary mask where all pixels corresponding to robot_id are set to 1, others to 0
        robot_mask = (object_id_mask == robot_id).astype(np.uint8)

    return rgb_image, depth_image, robot_mask


# Convert depth image to point cloud
def depth_to_point_cloud(
    depth_image, width, height, fov_v, min_depth=0.01, max_depth=2.9, robot_mask=None
):
    """
    Converts a depth image to a point cloud, filtering out pixels based on valid depth and the robot mask.

    Parameters:
        depth_image (numpy.ndarray): The depth image (2D array of depth values).
        width (int): The width of the depth image.
        height (int): The height of the depth image.
        fov_v (float): Field of view in degrees.
        min_depth (float): Minimum valid depth value (default: 0.1).
        max_depth (float): Maximum valid depth value (default: infinity).
        robot_mask (numpy.ndarray): A binary mask (same size as depth image) where pixels corresponding to the robot are 1.

    Returns:
        numpy.ndarray: A filtered point cloud (N x 3 array) with valid non-robot depth points.
    """
    # Intrinsics: Field of view
    fov_v_rad = np.radians(fov_v)
    fy = height / (2.0 * np.tan(fov_v_rad / 2.0))
    fx = fy  # Assuming aspect ratio = 1
    cx = width / 2.0
    cy = height / 2.0

    # Generate a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates
    x_camera = (x - cx) * depth_image / fx
    y_camera = (y - cy) * depth_image / fy
    z_camera = depth_image

    # Create a valid depth mask (filtering by min_depth and max_depth)
    valid_depth_mask = (depth_image > min_depth) & (depth_image < max_depth)

    if robot_mask is not None:
        # Combine valid depth mask with the inverse of the robot mask to exclude robot pixels
        valid_depth_mask = valid_depth_mask & (robot_mask == 0)

    # Apply the combined mask to filter the points
    x_camera_valid = x_camera[valid_depth_mask]
    y_camera_valid = y_camera[valid_depth_mask]
    z_camera_valid = z_camera[valid_depth_mask]

    # Stack the valid points into a point cloud (N x 3)
    points = np.vstack((x_camera_valid, y_camera_valid, z_camera_valid)).T

    return points


def draw_camera_frustum(
    camera_position,
    target_position,
    width=640,
    height=480,
    fov_v=57,
    near=0.01,
    far=10,
    life_time=0,
):
    aspect = width / height
    # Calculate the four corners of the near and far planes of the frustum
    near_height = 2 * np.tan(np.radians(fov_v) / 2) * near
    near_width = near_height * aspect
    far_height = 2 * np.tan(np.radians(fov_v) / 2) * far
    far_width = far_height * aspect

    # Get forward direction (camera to target) and normalize it
    forward_vector = np.array(target_position) - np.array(camera_position)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)

    # Right and up vectors for the camera
    up_vector = [0, 0, 1]  # Assuming z-up axis
    right_vector = np.cross(forward_vector, up_vector)
    up_vector = np.cross(right_vector, forward_vector)

    # Calculate centers of near and far planes
    near_center = np.array(camera_position) + forward_vector * near
    far_center = np.array(camera_position) + forward_vector * far

    # Calculate the four corners of the near and far planes
    near_top_left = (
        near_center + up_vector * (near_height / 2) - right_vector * (near_width / 2)
    )
    near_top_right = (
        near_center + up_vector * (near_height / 2) + right_vector * (near_width / 2)
    )
    near_bottom_left = (
        near_center - up_vector * (near_height / 2) - right_vector * (near_width / 2)
    )
    near_bottom_right = (
        near_center - up_vector * (near_height / 2) + right_vector * (near_width / 2)
    )

    far_top_left = (
        far_center + up_vector * (far_height / 2) - right_vector * (far_width / 2)
    )
    far_top_right = (
        far_center + up_vector * (far_height / 2) + right_vector * (far_width / 2)
    )
    far_bottom_left = (
        far_center - up_vector * (far_height / 2) - right_vector * (far_width / 2)
    )
    far_bottom_right = (
        far_center - up_vector * (far_height / 2) + right_vector * (far_width / 2)
    )

    color = [1, 0, 0]  # Red color for the wireframe
    line_width = 1

    # Draw the near and far plane edges
    p.addUserDebugLine(
        near_top_left, near_top_right, color, lineWidth=line_width, lifeTime=life_time
    )
    p.addUserDebugLine(
        near_top_right,
        near_bottom_right,
        color,
        lineWidth=line_width,
        lifeTime=life_time,
    )
    p.addUserDebugLine(
        near_bottom_right,
        near_bottom_left,
        color,
        lineWidth=line_width,
        lifeTime=life_time,
    )
    p.addUserDebugLine(
        near_bottom_left, near_top_left, color, lineWidth=line_width, lifeTime=life_time
    )

    p.addUserDebugLine(
        far_top_left, far_top_right, color, lineWidth=line_width, lifeTime=life_time
    )
    p.addUserDebugLine(
        far_top_right, far_bottom_right, color, lineWidth=line_width, lifeTime=life_time
    )
    p.addUserDebugLine(
        far_bottom_right,
        far_bottom_left,
        color,
        lineWidth=line_width,
        lifeTime=life_time,
    )
    p.addUserDebugLine(
        far_bottom_left, far_top_left, color, lineWidth=line_width, lifeTime=life_time
    )

    # Draw the lines connecting near and far planes (view cone lines)
    p.addUserDebugLine(
        far_top_left, near_top_left, color, lineWidth=line_width, lifeTime=life_time
    )
    p.addUserDebugLine(
        far_top_right, near_top_right, color, lineWidth=line_width, lifeTime=life_time
    )
    p.addUserDebugLine(
        far_bottom_right,
        near_bottom_left,
        color,
        lineWidth=line_width,
        lifeTime=life_time,
    )
    p.addUserDebugLine(
        far_bottom_left,
        near_bottom_right,
        color,
        lineWidth=line_width,
        lifeTime=life_time,
    )


def transform_point_cloud_to_world(cloud, camera_position, target_position):
    # Compute the view matrix

    camera_pose = calculate_camera_pose(camera_position, target_position)

    # print(camera_pose)

    draw_frame(camera_pose)

    # Convert the point cloud to homogeneous coordinates by adding a column of ones
    num_points = cloud.shape[0]
    homogeneous_cloud = np.hstack((cloud, np.ones((num_points, 1))))

    # Apply the view matrix transformation to all points in a batch operation
    transformed_cloud_homogeneous = np.dot(camera_pose, homogeneous_cloud.T).T

    # Convert back to 3D by discarding the homogeneous coordinate (4th column)
    transformed_cloud = transformed_cloud_homogeneous[:, :3]

    return transformed_cloud


def merge_point_clouds(
    cloud1, cloud2, camera1_pos, target_pos1, camera2_pos, target_pos2
):
    # Transform both clouds into the world frame
    transformed_cloud1 = transform_point_cloud_to_world(
        cloud1, camera1_pos, target_pos1
    )
    transformed_cloud2 = transform_point_cloud_to_world(
        cloud2, camera2_pos, target_pos2
    )

    # Merge the two transformed point clouds
    merged_cloud = np.vstack((transformed_cloud1, transformed_cloud2))

    return merged_cloud


def create_wall():
    # Wall parameters
    wall_width = 0.1
    wall_height = 2.0
    wall_length = 3.0

    # Create a cuboid as the wall
    wall_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_length / 2, wall_width / 2, wall_height / 2],
    )
    wall_visual_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_length / 2, wall_width / 2, wall_height / 2],
        rgbaColor=[0.6, 0.6, 0.6, 1],
    )

    # Position the wall 2 meters to the right of the robot along the x-axis
    wall_position = (0, 1.5, wall_height / 2)
    wall_id = p.createMultiBody(
        baseCollisionShapeIndex=wall_collision_id,
        baseVisualShapeIndex=wall_visual_id,
        basePosition=wall_position,
    )
    return wall_id



def create_scene1():

    p.connect(p.GUI)

    bd_path = pybullet_data.getDataPath() + "/"

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf")
    duck_id = p.loadURDF("duck_vhacd.urdf")
    shelf_id = p.loadSDF("kiva_shelf/model.sdf")[0]
    p.resetBasePositionAndOrientation(shelf_id, [0,-1.2,1.1],[0,0,0,1])

    wall_id = create_wall()

    urdf_file1 = "franka_panda/panda.urdf"
    base_position1 = (-0.4, 0, -0.0)
    robot1_id = p.loadURDF(urdf_file1, basePosition=base_position1, useFixedBase=True)

    initial_joint_positions1 = [
        -0.41,
        0.71,
        -0.00,
        -1.92,
        1.95,
        1.33,
        -3.33,
        0.0,
        0.0,
        0.01,
        0.01,
        0.0,
    ]
    for i in range(len(initial_joint_positions1)):
        p.resetJointState(robot1_id, i, initial_joint_positions1[i])

    for _ in range(3):
        create_random_sphere(0.05, 0.3)

    for _ in range(3):
        create_random_cuboid(0.05, 0.5)

    for _ in range(3):
        create_random_cylinder(0.05, 0.5)


    # Setup two cameras
    width = 640
    height = 480
    # width = 160
    # height = 120
    fov_v = 57
    near = 0.01
    far = 6
    camera1_pos = np.array([-2, 0, 1])  # 2 meters away from the left
    camera2_pos = np.array([2, 0, 1])  # 2 meters away from the right
    target_pos = np.array([0, 0, 0])  # Looking at the origin (robot)

    # Draw frustum for both cameras with a long life time
    draw_camera_frustum(
        camera1_pos, target_pos, width, height, fov_v, near, far, life_time=1000
    )
    draw_camera_frustum(
        camera2_pos, target_pos, width, height, fov_v, near, far, life_time=1000
    )

    merged_cloud = None  # Initialize the merged point cloud

    time_limit = 10
    start_time = time.time()
    while True:
        p.stepSimulation()

        # Add robot controller here

        # Capture depth images from both cameras at each step
        rgb_img1, depth_img1, robot_mask1 = capture_rgbd_image(
            camera1_pos, target_pos, width, height, fov_v, near, far, robot1_id
        )
        rgb_img2, depth_img2, robot_mask2 = capture_rgbd_image(
            camera2_pos, target_pos, width, height, fov_v, near, far, robot1_id
        )

        # Convert depth images to point clouds
        cloud1 = depth_to_point_cloud(
            depth_img1, width, height, fov_v, 0.01, 2.9, robot_mask1
        )
        cloud2 = depth_to_point_cloud(
            depth_img2, width, height, fov_v, 0.01, 2.9, robot_mask2
        )

        # Merge point clouds with transformation to the world frame
        merged_cloud = merge_point_clouds(
            cloud1, cloud2, camera1_pos, target_pos, camera2_pos, target_pos
        )

        merged_pcd = voxel_downsample(merged_cloud, 0.01)

        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            break

    # visualize_point_cloud(merged_cloud)

    save_depth_image(depth_img1,near,far,"depth_1.bmp")
    save_depth_image(depth_img2,near,far,"depth_2.bmp")

    
    return merged_pcd,[rgb_img1,rgb_img2],[depth_img1,depth_img2]


# Example usage
if __name__ == "__main__":

    merged_pcd,rgb_images,depth_images = create_scene1()

    for i,a in enumerate(rgb_images):
        img = Image.fromarray(a)
        img.save('rgb_'+str(i)+'.bmp')


    o3d.io.write_point_cloud("scene.pcd", merged_pcd)
    o3d.visualization.draw_geometries([merged_pcd])
