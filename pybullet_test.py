import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf", )
startPos = [0,0,0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robot_id = p.loadURDF("fobo2.xacro",startPos, startOrientation)

# for i in range(p.getNumJoints(bodyUniqueId = robot_id)):
#     print(i)
#     print(p.getJointInfo(bodyUniqueId = robot_id, jointIndex = i))

while True:
    p.stepSimulation()
    time.sleep(1./240.)

# #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
# for i in range (10000):
#     p.getCameraImage(224, 224, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     p.stepSimulation()
#     p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=2, controlMode=p.VELOCITY_CONTROL, targetVelocity=-50)
#     p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=3, controlMode=p.VELOCITY_CONTROL, targetVelocity=-50)
#     time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(robot_id)
# print(cubePos,cubeOrn)
p.disconnect()
