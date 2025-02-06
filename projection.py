import numpy as np
import imageio.v2 as imageio
import argparse
import os
import cv2 as cv


class Projection:
    def __init__(self, width=640, height=480, fov=79):
        self.camera_intrinsic = None
        self.width = width  # 图像的宽
        self.height = height  # 图像的高
        self.camera_matrix = None
        self.get_camera_matrix(width, height, fov)

    def get_camera_matrix(self, width, height, fov):  # 这里是获取相机的内参，后面如果要实际运行就得自己改
        """Returns a camera matrix from image size and fov."""
        # 计算光学中心
        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.
        # 计算焦距
        f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
        self.camera_intrinsic = {'cx': cx, 'cy': cy, 'f': f}
        self.camera_matrix = np.array([[f, 0.0, cx],
                                       [0.0, f, cy],
                                       [0, 0, 1]], dtype=np.float32)

    def projection(self, rgb, depth, semantic, transform_height, instance=None, num_class=22):
        j_indices, i_indices = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='xy')

        valid_mask = (depth > 0)
        d = depth[valid_mask]
        p = np.vstack((j_indices[valid_mask] * d, i_indices[valid_mask] * d, d))

        camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        point_1 = np.dot(camera_matrix_inv, p)
        point_1[1] += transform_height

        point_2 = np.dot(self.camera_matrix, point_1)
        point_2 /= point_2[2]

        rgb_t = np.zeros_like(rgb)
        depth_t = np.zeros_like(depth)
        sem_t = np.zeros_like(semantic) + num_class
        instance_t = None
        if instance is not None:
            instance_t = np.zeros_like(instance)

        in_bounds = (point_2[0] >= 0) & (point_2[0] < self.width) & \
                    (point_2[1] >= 0) & (point_2[1] < self.height)

        i_proj, j_proj = point_2[1][in_bounds].astype(int), point_2[0][in_bounds].astype(int)
        i_orig, j_orig = i_indices[valid_mask][in_bounds], j_indices[valid_mask][in_bounds]

        rgb_t[i_proj, j_proj] = rgb[i_orig, j_orig]
        depth_t[i_proj, j_proj] = depth[i_orig, j_orig]
        sem_t[i_proj, j_proj] = semantic[i_orig, j_orig]
        if instance is not None:
            instance_t[i_proj, j_proj] = instance[i_orig, j_orig]
            return rgb_t.astype(np.uint8), depth_t / 10.0, sem_t.astype(np.uint8), instance_t.astype(
                np.uint16)  # 返回的深度要重新转换为千米

        return rgb_t.astype(np.uint8), depth_t / 10.0, sem_t.astype(np.uint8)  # 返回的深度要重新转换为千米


def viewpointTransform(args):
    # 设置数据保存路径
    root_path = os.path.join('./Semantic_Dataset/labeled/',
                             str(args.source_height) + "->" + str(args.target_height) + "_projection")
    rgb_path = os.path.join(root_path, 'rgb')
    depth_path = os.path.join(root_path, 'depth')
    semantic_path = os.path.join(root_path, 'semantic')
    instance_path = os.path.join(root_path, 'instance')
    if not os.path.exists(root_path):  # 创建文件路径
        os.makedirs(rgb_path)
        os.makedirs(depth_path)
        os.makedirs(semantic_path)
        os.makedirs(instance_path)

    # 设置读取图像的路径
    read_path = os.path.join('./Semantic_Dataset/labeled/', str(args.source_height))
    read_rgb_path = os.path.join(read_path, 'rgb')
    read_depth_path = os.path.join(read_path, 'depth')
    read_semantic_path = os.path.join(read_path, 'semantic')
    read_instance_path = os.path.join(read_path, 'instance')

    # 将0.68的图像投影到其它高度
    total_image = len(os.listdir(rgb_path))
    projection = Projection()

    for image_id in range(0, total_image):
        if image_id % 20 == 0:
            print("当前为", image_id)
        str_name = "{:07d}".format(image_id)

        rgb_file = os.path.join(read_rgb_path, str_name + '.jpg')
        depth_file = os.path.join(read_depth_path, str_name + '.npy')
        sem_file = os.path.join(read_semantic_path, str_name + '.png')
        instance_file = os.path.join(read_instance_path, str_name + '.png')

        rgb = cv.imread(rgb_file)
        depth = np.load(depth_file)  # 保存时用的是千米，所以后面投影时需要调整单位
        sem = imageio.imread(sem_file)
        instance = imageio.imread(instance_file)

        # 进行投影
        rgb_t, depth_t, sem_t, instance_t = projection.projection(rgb=rgb, depth=depth * 10.0, semantic=sem,
                                                                  transform_height=args.target_height - args.source_height,
                                                                  instance=instance)  # 注意：这里的深度要转换为米再输入

        # 保存
        rgb_file = os.path.join(rgb_path, str_name + '.jpg')
        depth_file = os.path.join(depth_path, str_name + '.npy')
        sem_file = os.path.join(semantic_path, str_name + '.png')
        ins_file = os.path.join(instance_path, str_name + '.png')

        cv.imwrite(rgb_file, rgb_t)
        np.save(depth_file, depth_t)
        cv.imwrite(sem_file, sem_t)
        cv.imwrite(ins_file, instance_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Projection')
    parser.add_argument('--source_height', type=float, default=0.88, help='source height (meter)')
    parser.add_argument('--target_height', type=float, default=0.28, help='target height (meter)')
    args = parser.parse_args()
    viewpointTransform(args)
