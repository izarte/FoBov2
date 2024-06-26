import os
import numpy as np
import pybullet as p
import cv2

from fobo2_env.src.utils import (
    random_pos_orientation,
    rotate_by_yaw,
    orientation_in_grad,
    position_in_range
)


class Robot:
    class Camera:
        def __init__(
            self,
            client_id: int,
            fov: int = 60,
            near: float = 2,
            far: float = 40,
            width: int = 128,
            height: int = 128,
            relative_pose: list = [0.2, 0, 0.15],
            relative_orientation: list = [0, 0, 0],
            mode: str = "rgb",
        ):
            self.client_id = client_id
            self.params = {
                "fov": fov,
                "near": near,
                "far": far,
                "width": width,
                "height": height,
                "aspect": width / height,
            }
            self.mode = mode
            self.relative_pose = np.array(relative_pose)
            self.relative_orientation = relative_orientation
            self.current_pos = [0, 0, 0]
            self.current_orientation = {"roll": 0, "pitch": 0, "yaw": 0}

        def reset(self, robot_pos):
            self.current_pos = robot_pos
            self.current_orientation = {"roll": 0, "pitch": 0, "yaw": 0}

        def get_image(self):
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                physicsClientId=self.client_id,
                cameraTargetPosition=self.current_pos,
                distance=0.25,
                roll=self.current_orientation["roll"] + self.relative_orientation[0],
                pitch=self.current_orientation["pitch"] + self.relative_orientation[1],
                yaw=self.current_orientation["yaw"] + self.relative_orientation[2],
                upAxisIndex=2,
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                physicsClientId=self.client_id,
                fov=self.params["fov"],
                aspect=self.params["aspect"],
                nearVal=self.params["near"],
                farVal=self.params["far"],
            )

            if self.mode == "rgb":
                _, _, rgb, _, _ = p.getCameraImage(
                    physicsClientId=self.client_id,
                    width=self.params["width"],
                    height=self.params["height"],
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    shadow=True,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=p.ER_NO_SEGMENTATION_MASK,
                )
                # image = np.reshape(rgb, (self.params["height"], self.params["width"], 3))
                rgb = rgb[:, :, :3]
                image = rgb
                image = image.astype(np.uint8)
            elif self.mode == "depth":
                _, _, _, depth, _ = p.getCameraImage(
                    physicsClientId=self.client_id,
                    width=self.params["width"],
                    height=self.params["height"],
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    shadow=True,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=p.ER_NO_SEGMENTATION_MASK,
                )
                depth_buffer = np.reshape(
                    depth, [self.params["width"], self.params["height"]]
                )
                far = self.params["far"]
                near = self.params["near"]
                normalized_depth = normalized_depth = (depth_buffer - near) / (
                    far - near
                )
                image = (normalized_depth * 255).astype(np.uint8)
            return image

        def add_noise(self, image):
            mean = 0
            stddev = 50
            noise = np.zeros_like(image, dtype=np.float32)
            cv2.randn(noise, mean, stddev)
            noisy_image = cv2.add(image.astype(np.float32), noise)
            return np.clip(noisy_image, 0, 255).astype(np.uint8)

        def move(self, robot_position: np.array, roll: float, pitch: float, yaw: float):
            rotate_relative_pos = rotate_by_yaw(self.relative_pose, yaw)
            self.current_pos = robot_position + rotate_relative_pos
            self.current_orientation["roll"] = roll
            self.current_orientation["pitch"] = pitch
            self.current_orientation["yaw"] = yaw

    def __init__(
        self,
        client_id: int,
        depth_width: int = 128,
        depth_height: int = 128,
        rgb_width: int = 128,
        rgb_height: int = 128,
    ):
        self.right_wheel = 1
        self.left_wheel = 3
        self.client_id = client_id
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.id = 0

        self.depth_camera = self.Camera(
            client_id=client_id,
            fov=70,
            near=1,
            far=20,
            width=self.depth_width,
            height=self.depth_height,
            relative_pose=[0.50, 0, 0.6],
            relative_orientation=[0, 0, 0],
            mode="depth",
        )

        self.rgb_camera = self.Camera(
            client_id=client_id,
            fov=84,
            near=0.02,
            far=10000,
            width=self.rgb_width,
            height=self.rgb_height,
            relative_pose=[0.50, 0, 0.15],
            relative_orientation=[0, 30, 0],
            mode="rgb",
        )

    def reset_human_based(self, human_pose, area, desired_distance):
        r = np.random.uniform(desired_distance - 0.2, desired_distance + 0.2)
        while True:
            angle = np.random.random() * np.pi
            robot_start_pos = [human_pose[0] + (r * np.cos(angle)), human_pose[1] + (r * np.sin(angle))]
            if position_in_range(position=robot_start_pos, area=area):
                break

        yaw_to_human = np.arctan2(
            robot_start_pos[1] - human_pose[1],
            robot_start_pos[0] - human_pose[0],
        )

        yaw_to_human += np.random.uniform(-np.radians(45), np.radians(45))
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, yaw_to_human])

        
        self.id = p.loadURDF(
            physicsClientId=self.client_id,
            fileName=os.path.dirname(__file__) + "/models/fobo2.urdf",
            basePosition=[robot_start_pos[0], robot_start_pos[1], 0.25],
            baseOrientation=robot_start_orientation,
            useFixedBase=False,
        )
        self.depth_camera.reset(robot_start_pos)
        self.rgb_camera.reset(robot_start_pos)
        self.move([0, 0])

        

    def reset(self, starting_area):
        robot_start_pos, robot_start_orientation = random_pos_orientation(
            (starting_area[0], starting_area[1], 0.3)
        )
        self.id = p.loadURDF(
            physicsClientId=self.client_id,
            fileName=os.path.dirname(__file__) + "/models/fobo2.urdf",
            basePosition=robot_start_pos,
            baseOrientation=robot_start_orientation,
            useFixedBase=False,
        )
        self.depth_camera.reset(robot_start_pos)
        self.rgb_camera.reset(robot_start_pos)
        self.move([0, 0])

    def move(self, action):
        # Right wheel
        p.setJointMotorControl2(
            physicsClientId=self.client_id,
            bodyIndex=self.id,
            jointIndex=self.right_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=action[1],
        )
        # Left wheel
        p.setJointMotorControl2(
            physicsClientId=self.client_id,
            bodyIndex=self.id,
            jointIndex=self.left_wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=action[0],
        )
        robot_pose, robot_orientation = p.getBasePositionAndOrientation(
            physicsClientId=self.client_id, bodyUniqueId=self.id
        )
        roll, pitch, yaw = orientation_in_grad(
            p.getEulerFromQuaternion(quaternion=robot_orientation)
        )

        # Calculate camera positions
        self.depth_camera.move(
            robot_position=np.array(robot_pose), roll=roll, pitch=pitch, yaw=yaw
        )

        self.rgb_camera.move(
            robot_position=np.array(robot_pose), roll=roll, pitch=pitch, yaw=yaw
        )

    def get_images(self):
        depth_image = self.depth_camera.get_image()
        rgb_image = self.rgb_camera.get_image()
        noise_depth = self.depth_camera.add_noise(depth_image)
        noise_rgb = self.rgb_camera.add_noise(rgb_image)
        return noise_rgb, noise_depth

    def get_motor_speeds(self):
        _, speedL, _, _ = p.getJointState(
            physicsClientId=self.client_id,
            bodyUniqueId=self.id,
            jointIndex=self.left_wheel,
        )
        _, speedR, _, _ = p.getJointState(
            physicsClientId=self.client_id,
            bodyUniqueId=self.id,
            jointIndex=self.right_wheel,
        )
        return speedL, speedR

    def get_human_coordinates(self, rgb):
        return 0, 0
