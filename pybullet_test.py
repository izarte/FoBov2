import time

import numpy as np
import pybullet as p
import pybullet_data

from walking_functions import get_all_y_uniform


def to_rad(n):
    return n * 3.1416 / 180


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, 0)
# p.setGravity(0, 0, -9.81)
planeId = p.loadURDF(
    "plane.urdf",
)
startPos = [0, 0, 1.1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
# robot_id = p.loadURDF("urdf/fobo2.urdf",startPos, startOrientation)
human_id = p.loadURDF("urdf/human.urdf", startPos, startOrientation)

# for i in range(p.getNumJoints(bodyUniqueId = human_id)):
#     print(i)
#     print(p.getJointInfo(bodyUniqueId = human_id, jointIndex = i))

total_steps = 240 * 3
uniform_ankle, uniform_knee, uniform_hip = get_all_y_uniform(total_steps)
print(uniform_ankle[0])
right_hip = 33
right_knee = 35
right_ankle = 38

left_hip = 42
left_knee = 44
left_ankle = 47

i = total_steps
j = total_steps // 2
while True:
    p.stepSimulation()
    position, quaternion = p.getBasePositionAndOrientation(bodyUniqueId=human_id)

    new_position = np.array(position) + np.array([0.001, 0, 0])

    p.resetBasePositionAndOrientation(
        bodyUniqueId=human_id, posObj=new_position, ornObj=startOrientation
    )
    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=right_hip,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_hip[i]),
    )
    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=right_knee,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_knee[i]),
    )
    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=right_ankle,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_ankle[i]),
    )

    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=left_hip,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_hip[j]),
    )
    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=left_knee,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_knee[j]),
    )
    p.setJointMotorControl2(
        bodyIndex=human_id,
        jointIndex=left_ankle,
        controlMode=p.POSITION_CONTROL,
        targetPosition=to_rad(uniform_ankle[j]),
    )

    time.sleep(1.0 / 240.0)
    i -= 1
    j -= 1
    if i == -1:
        i = total_steps
    if j == -1:
        j = total_steps

# width = 128
# height = 128

# fov = 60
# aspect = width / height
# near = 0.02
# far = 1

# def to_grad(n):
#     return n * 180 / 3.1416

# # #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
# for i in range (10000):
#     # p.getCameraImage(224, 224, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     p.stepSimulation()
#     position, quaternion = p.getBasePositionAndOrientation(bodyUniqueId=robot_id)
#     position = list(position)
#     position[2] += 0.2

#     # # view_matrix = p.computeViewMatrix(position, [0, 0, 0], [1, 0, 0])
#     roll, pitch, yaw = p.getEulerFromQuaternion(quaternion=quaternion)
#     # yaw += 1.5707
#     # print(roll, pitch, yaw)
#     view_matrix = p.computeViewMatrixFromYawPitchRoll(
#         cameraTargetPosition=position,
#         distance = 0.25,
#         roll = to_grad(roll),
#         pitch = to_grad(pitch),
#         yaw = to_grad(yaw) + 90,
#         upAxisIndex = 2
#     )
#     projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
#     images = p.getCameraImage(width,
#                             height,
#                             view_matrix,
#                             projection_matrix,
#                             shadow=True,
#                             renderer=p.ER_BULLET_HARDWARE_OPENGL,
#                             flags=p.ER_NO_SEGMENTATION_MASK)
#     rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
#     depth_buffer_opengl = np.reshape(images[3], [width, height])
#     depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

#     p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=5)
#     p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=3, controlMode=p.VELOCITY_CONTROL, targetVelocity=-5)
#     time.sleep(1./240.)
# plt.subplot(4, 2, 1)
# plt.imshow(depth_opengl, cmap='gray', vmin=0, vmax=1)
# plt.show()
# cubePos, cubeOrn = p.getBasePositionAndOrientation(robot_id)
# print(cubePos,cubeOrn)
p.disconnect()
