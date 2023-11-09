import numpy as np
import open3d as o3d

def visualize_rgbd(rgbd_image):
    # print(rgbd_image)

    # o3d.visualization.draw_geometries([rgbd_image])
    print(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print()
    # print(o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

def get_rgbd(color: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
    try:
        o3d_color = o3d.geometry.Image(color)
    except:
        o3d_color = o3d.geometry.Image(np.array(color))

    try:
        o3d_depth = o3d.geometry.Image(depth)
    except:
        o3d_depth = o3d.geometry.Image(np.array(depth))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth,
                                                              convert_rgb_to_intensity=False)
    return rgbd

def get_pcd(rgbd: o3d.geometry.RGBDImage) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

def rgbd2pcd(color: np.ndarray, depth: np.ndarray = None) -> o3d.geometry.PointCloud:
    if color.shape[-1] == 3 and depth is not None:
        rgbd = get_rgbd(color, depth)

    elif color.shape[-1] == 4 and depth is None:
        rgbd = get_rgbd(color[:, :, :3].astype(np.uint8), color[:, :, -1])
    else:
        raise RuntimeError('color argument is must be 3ch RGB or 4ch RGBD')

    pcd = get_pcd(rgbd)

    return pcd

def get_down_pcd(pcd: o3d.geometry.PointCloud, size: float = 0.01) -> o3d.geometry.PointCloud:
    down_pcd = pcd.voxel_down_sample(voxel_size=size)
    return down_pcd

def get_norm(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01, x: int = None, y: int = None):
    pcd = get_down_pcd(pcd, voxel_size)

    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()
    return pcd

def get_convexhull(pcd: o3d.geometry.PointCloud):
    # 포인트 클라우드에 대한 컨벡스 헐을 계산
    hull, _ = pcd.compute_convex_hull()

    # 컨벡스 헐을 메시(mesh)로 변환
    hull_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(hull.vertices),
                                          triangles=o3d.utility.Vector3iVector(hull.triangles))

    # 메시를 라인셋으로 변환하여 각각의 에지를 볼 수 있도록 함
    hull_lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull_mesh)

    return hull_lines


