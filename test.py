import cv2
import numpy as np
import open3d as o3d
import torch
from ketisdk.sensor.realsense_sensor import RSSensor
import src.norm_vector as nv


PCD = False

def main():
    sensor = RSSensor()
    sensor.start()
    sensor.get_data()

    intrinsics = [sensor.info.fx, sensor.info.fy, sensor.info.cx, sensor.info.cy]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    midas_transforms = midas_transforms.dpt_transform

    midas.to(device)
    midas.eval()

    if PCD:
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=720, width=1280)

        pcd = o3d.geometry.PointCloud()
        points = np.random.rand(10, 3)
        pcd.points = o3d.utility.Vector3dVector(points)

        vis.add_geometry(pcd)

    while True:
        color, depth = sensor.get_data()

        midas_input = midas_transforms(color).to(device)

        with torch.no_grad():
            midas_depth = midas(midas_input)
            midas_depth = torch.nn.functional.interpolate(midas_depth.unsqueeze(1),
                                                          size=color.shape[:2],
                                                          mode='bilinear',
                                                          align_corners=False).squeeze()
        midas_depth = midas_depth.cpu().numpy()

        print(midas_depth.dtype, midas_depth.shape)
        print(np.mean(midas_depth))


        if PCD:
            pcd = cvt_pcd(pcd, color, depth)

            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

    # vis.destroy_window()

def cvt_pcd(pcd, color, depth):
    new_pcd = nv.rgbd2pcd(color, depth)
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors

    return pcd


if __name__ == '__main__':
    main()