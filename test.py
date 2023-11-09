import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import open3d as o3d
import torch
from ketisdk.sensor.realsense_sensor import RSSensor
import src.norm_vector as nv
from sklearn.linear_model import LinearRegression

PCD = False


# 예제로 사용할 가상의 2D depth 데이터를 생성합니다.
depth_data = np.random.randint(0, 256, size=(480, 640), dtype=np.uint8)

# 마우스 커서 위치의 depth 값을 저장할 전역 변수입니다.
depth_value_at_cursor = None

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         # 마우스 커서 위치의 depth 값을 가져옵니다.
#
#         x_len = int(depth_data.shape[1] / 2)
#         depth_value1 = depth_data[y, x]
#         try:
#             depth_value2 = depth_data[y, x + x_len]
#             print(f"Depth at position [{x},\t{y}]: {depth_value1}   -   {depth_value2}")
#         except:
#             depth_value2 = depth_data[y, x - x_len]
#             print(f"Depth at position [{x},\t{y}]: {depth_value2}   -   {depth_value1}")
#
# # 'Depth Display'라는 이름의 윈도우를 생성합니다.
# cv2.namedWindow('Depth Display')
#
# # 마우스 콜백 함수를 설정합니다.
# cv2.setMouseCallback('Depth Display', mouse_callback)

def map_values_2d(array, source_min, source_max, target_min, target_max):
    # 2D 배열의 각 값을 새로운 범위로 선형 변환합니다.
    return target_min + ((array - source_min) * (target_max - target_min)) / (source_max - source_min)

def main():
    global depth_data

    sensor = RSSensor()
    sensor.start()
    sensor.get_data()

    intrinsics = [sensor.info.fx, sensor.info.fy, sensor.info.cx, sensor.info.cy]

    scale_model = LinearRegression()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    midas_transforms = midas_transforms.dpt_transform

    midas.to(device)
    midas.eval()

    matching_index = [[66, 92],
                      [67, 180],
                      [82, 307],
                      [117, 892],
                      # [106, 1048],
                      # [104, 1102],
                      [336, 80],
                      [517, 167],
                      [466, 634],
                      [460, 894],
                      # [537, 1038],
                      [674, 643],
                      # [653, 1104]
                      ]

    if PCD:
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=720, width=1280)

        pcd = o3d.geometry.PointCloud()
        points = np.random.rand(10, 3)
        pcd.points = o3d.utility.Vector3dVector(points)

        vis.add_geometry(pcd)

    while True:
        color, depth = sensor.get_data()
        color = color[:, 200:, :]
        depth = depth[:, 200:]
        depth = np.clip(depth, 0, 1000)

        midas_input = midas_transforms(color).to(device)

        with torch.no_grad():
            midas_depth = midas(midas_input)
            midas_depth = torch.nn.functional.interpolate(midas_depth.unsqueeze(1),
                                                          size=color.shape[:2],
                                                          mode='bilinear',
                                                          align_corners=False).squeeze()
        midas_depth = midas_depth.cpu().numpy()
        midas_depth_inv = np.max(midas_depth) - midas_depth
        # print(np.min(midas_depth_inv), np.max(midas_depth_inv))
        # midas_depth_inv *= 100

        midas_depth_inv = midas_depth_inv / np.max(midas_depth_inv)     # 0 ~ 1
        # midas_depth_inv = midas_depth_inv / 14  # 0 ~ 1
        # midas_depth_inv = np.clip(midas_depth_inv, 0, 1)
        midas_depth_inv = midas_depth_inv * 2 ** 16                     # 0 ~ 2**16
        # midas_depth_inv = midas_depth_inv.astype(np.uint16)

        # rel_depth_values = np.array([midas_depth_inv[y, x] for y, x in matching_index])
        # abs_depth_values = np.array([depth[y, x] for y, x in matching_index])
        #
        # scale_model.fit(rel_depth_values.reshape(-1, 1), abs_depth_values)
        #
        # scaling_factor = scale_model.coef_[0]
        # print(scaling_factor)
        # # print(scaling_factor)
        # midas_abs = midas_depth_inv * scaling_factor
        # # print(np.min(midas_abs), np.max(midas_abs), np.mean(midas_abs))

        # midas_depth_inv = midas_depth_inv * 2 ** 16
        # midas_abs = midas_abs.astype(np.uint16)

        # midas_depth_inv = np.clip(midas_depth_inv, 0, 65535)
        # midas_depth_inv = midas_depth_inv.astype(np.uint16)
        # midas_depth = np.clip(midas_depth, 0.4, 2.0)
        # print(midas_depth.dtype, midas_depth.shape)
        # print(np.mean(midas_depth))


        # depth_data = midas_depth

        #
        # source_min = 20000
        # source_max = 40000
        #
        # target_min = 10000
        # target_max = 50000

        # midas_depth_inv = map_values_2d(midas_depth_inv, source_min, source_max, target_min, target_max)
        # midas_depth_inv = np.clip(midas_depth_inv, target_min, target_max)
        midas_depth_inv = midas_depth_inv.astype(np.uint16)

        target_min = 20000
        target_max = 40000

        source_min = 27100 / 50
        source_max = 32500 / 50

        print(np.min(depth), np.max(depth), np.mean(depth))
        depth = map_values_2d(depth, source_min, source_max, target_min, target_max)
        print(np.min(depth), np.max(depth), np.mean(depth))
        print()
        depth = np.clip(depth, 0, 2**16-1)
        depth = depth.astype(np.uint16)



        depth_data = np.hstack((depth, midas_depth_inv))

        # depth_data = np.vstack((color2, depth_data))

        # 이미지와 텍스트를 표시합니다.
        cv2.imshow('Depth Display', depth_data)
        cv2.imshow('Depth Display', color[:, :, ::-1])

        # 1ms 동안 키 입력을 기다립니다. 'q'를 누르면 루프에서 빠져나옵니다.
        cv2.waitKey(0)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break




        if PCD:
            color = color
            depth = midas_depth_inv

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