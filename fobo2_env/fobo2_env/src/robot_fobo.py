import pybullet as p
import numpy as np
import os

from fobo2_env.src.utils import random_pos_orientation


class Robot():
    class Camera():
        def __init__(
                self,
                client_id : int,
                fov : int = 60,
                near : float = 0.02,
                far : float = 4,
                width : int = 128,
                height :int = 128,
                relative_pose : list = [0.2, 0, 0.15],
                relative_orientation : list = [0,0,0],
                mode : str = "rgb"
            ):
            self.client_id = client_id
            self.params = {
                "fov": fov,
                "near": near,
                "far": far,
                "width": width,
                "height": height,
                "aspect": width / height
            }
            self.mode = mode
            self.relative_pose = np.array(relative_pose)
            self.relative_orientation = p.getQuaternionFromEuler(relative_orientation)
            self.current_pos = [0, 0, 0]
            self.current_orientation = {"roll": 0, "pitch": 0, "yaw": 0}
            
        def reset(self):
            self.current_pos = [0, 0, 0]
            self.current_orientation = {"roll": 0, "pitch": 0, "yaw": 0}

        def get_image(self):
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                physicsClientId = self.client_id,
                cameraTargetPosition = self.current_pos,
                distance = 0.25,
                roll = self.current_orientation["roll"],
                pitch = self.current_orientation["pitch"],
                yaw = self.current_orientation["yaw"],
                upAxisIndex = 2
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                physicsClientId = self.client_id,
                fov = self.params["fov"],
                aspect = self.params["aspect"],
                nearVal = self.params["near"],
                farVal = self.params["far"]
            )
            # images = p.getCameraImage(
            #     physicsClientId = self.client_id,
            #     width = self.params["width"],
            #     height = self.params["height"],
            #     viewMatrix = view_matrix,
            #     projectionMatrix = projection_matrix,
            #     shadow=True,
            #     renderer=p.ER_BULLET_HARDWARE_OPENGL,
            #     flags=p.ER_NO_SEGMENTATION_MASK
            # )
            # if self.mode == 'rgb':
            #     image = np.reshape(images[2], (self.params["height"], self.params["width"], 4)) * 1. / 255.
            # elif self.mode == 'depth':
            #     depth_buffer_opengl = np.reshape(images[3], [self.params["width"], self.params["height"]])
            #     far = self.params["far"]
            #     near = self.params["near"]
            #     image = far * near / (far - (far - near) * depth_buffer_opengl)

            if self.mode == 'rgb':
                _, _, rgb, _, _ = p.getCameraImage(
                    physicsClientId = self.client_id,
                    width = self.params["width"],
                    height = self.params["height"],
                    viewMatrix = view_matrix,
                    projectionMatrix = projection_matrix,
                    shadow=True,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=p.ER_NO_SEGMENTATION_MASK
                )
                image = np.reshape(rgb, (self.params["height"], self.params["width"], 4)) * 1. / 255.
            elif self.mode == 'depth':
                _, _, _, depth, _ = p.getCameraImage(
                    physicsClientId = self.client_id,
                    width = self.params["width"],
                    height = self.params["height"],
                    viewMatrix = view_matrix,
                    projectionMatrix = projection_matrix,
                    shadow=True,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=p.ER_NO_SEGMENTATION_MASK
                )
                depth_buffer_opengl = np.reshape(depth, [self.params["width"], self.params["height"]])
                far = self.params["far"]
                near = self.params["near"]
                image = far * near / (far - (far - near) * depth_buffer_opengl)

            return image
        
        # TODO add noise function to image
        def add_noise(self, image):
            return image

        def move(
                self,
                robot_position : np.array,
                roll : float,
                pitch : float,
                yaw : float
            ):
            rotate_relative_pos = self.rotate_by_yaw(robot_position, yaw)
            self.current_pos = robot_position + rotate_relative_pos
            self.current_orientation["roll"] = roll
            self.current_orientation["pitch"] = pitch
            self.current_orientation["yaw"] = yaw

        def rotate_by_yaw(
                self,
                position : np.array,
                yaw : float
            ):
            r_pos = np.copy(position)
            yaw = np.radians(yaw)
            r_pos[0] = -np.sin(yaw) * position[0]
            r_pos[1] = np.cos(yaw) * position[0]
            return r_pos 

    def __init__(
            self,
            client_id : int,
            depth_width : int = 128,
            depth_height : int = 128,
            rgb_width : int = 128,
            rgb_height : int = 128
        ):

        self.client_id = client_id
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        self.depth_camera = self.Camera(
            client_id = client_id,
            fov = 60,
            near = 0.02,
            far = 4,
            width = self.depth_width,
            height = self.depth_height,
            relative_pose = [0.2, 0, 0.15],
            relative_orientation = [0, 0, 0],
            mode = 'depth'
        )

        self.rgb_camera = self.Camera(
            client_id = client_id,
            fov = 60,
            near = 0.02,
            far = 4,
            width = self.depth_width,
            height = self.depth_height,
            relative_pose = [0.2, 0, 0.15],
            relative_orientation = [0, 0, 0],
            mode = 'rgb'
        )
        self.robot_id = 0


    def reset(self):
        robot_start_pos, robot_start_orientation = random_pos_orientation()

        self.robot_id = p.loadURDF(
            physicsClientId = self.client_id,
            fileName = os.path.dirname(__file__) + "/models/fobo2.urdf", 
            basePosition = robot_start_pos,
            baseOrientation = robot_start_orientation,
            useFixedBase = False
        )
        self.depth_camera.reset()
        self.rgb_camera.reset()

        return self.robot_id
    
    def move(self, action):
        p.setJointMotorControl2(
            physicsClientId = self.client_id,
            bodyIndex=self.robot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=action[0]
        )
        # Right wheel
        p.setJointMotorControl2(
            physicsClientId = self.client_id,
            bodyIndex=self.robot_id,
            jointIndex=3,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=action[1]
        )
        robot_pose, robot_orientation = p.getBasePositionAndOrientation(
            physicsClientId = self.client_id,
            bodyUniqueId = self.robot_id
            )
        roll, pitch, yaw = self.orientation_in_grad(p.getEulerFromQuaternion(quaternion=robot_orientation))

        # Calculate camera positions
        self.depth_camera.move(
            robot_position = np.array(robot_pose),
            roll = roll,
            pitch = pitch,
            yaw = yaw
        )

        self.rgb_camera.move(
            robot_position = np.array(robot_pose),
            roll = roll,
            pitch = pitch,
            yaw = yaw
        )

    def get_images(self):
        depth_image = self.depth_camera.get_image()
        rgb_image = self.rgb_camera.get_image()
        noise_depth = self.depth_camera.add_noise(depth_image)
        noise_rgb = self.rgb_camera.add_noise(rgb_image)

        return noise_rgb, noise_depth

    def orientation_in_grad(self, orientation):
        roll = np.degrees(orientation[0])
        pitch = np.degrees(orientation[1])
        yaw = np.degrees(orientation[2])

        return roll, pitch, yaw + 90
