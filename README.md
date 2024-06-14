# Fobov2
This project is the final master's work in artificial intelligence. This is the second version of [FOBO].
Fobo is a robot whose main task is to follow you to carry your heavy weights.

This version has one fixed ToF camera (RGBD), with it, the following person is detected using MobileNetV2 and tracker with SORT (code found in forked repo [mobile-net-for-fobo]) and movement and avoiding object logic is trained using reinforcement learning.

![robot]

Train simulation video can be found in this link <a href="https://youtu.be/na0y2CK5VwM" target="_blank">demo</a>. The real application can be seen in this video <a href="https://youtu.be/sEIGYYCS6TI" target="_blank">real-demo</a>




## Reinforcement learning
The environment is defined in [Gymnasium] integrated with the physics simulator [PyBullet]. The reinforcement learning algorithms used are SAC (Soft Actor-Critic) and PPO (Proximal Policy Optimization) giving better results with SAC.

Model inputs are:
1. Depth image normalized in [0, 1]
2. Human centroid in image pixels normalized [-1, 1] (-1 when the human is not found in image)
3. Motors speed normalized in the range [-1, 1]

Model outputs are:
1.  Left wheel speed [-1, 1]
2.  Right wheel speed [-1, 1]

To train a model it is recommended to use docker but it can be trained by defining environment variables found in [train_fobo.py]. To use train using docker modify parameters and volumes in [docker-compose-train.yaml] and execute:

```bash
docker compose up -d
```

Hyperparameters for both training algorithms can be optimized using [Optuna] framework. Like training it can be done with docker in [docker-compose-hyperparameters.yaml] to select an algorithm, volumes and GPU usage or locally with environment variable set up.

Likewise, previous usages, the evaluation of models can be done with docker [docker-compose-eval.yaml].

## Hardware
To create this robot it is necessary:
 1. Raspberry Pi 4 model B 4GB
 2. ESP32
 3. Google Coral USB Accelerator
 4. Arducam ToF
 5. Aukey Webcam FullHD USB
 6. 2x Chihai-motor CHP-36GP-555-ABHL
 7. 2x Metallic steel ball caster wheel
 8. 20x Ni-MH cells
 9. Dual H-bridge DC motor controller
 10. 2x Fan 5v

The following diagram shows how devices are connected

![circuit]

## Instalation and real setup

First installation must be done following [raspberry-installation] steps and docker.

Download the forked repository for mobilenet with git submodules

```bash
git submodule init 
git submodule update
```

Go to [src] folder and execute the following command to launch all containers needed for local execution 
```bash
docker compose up -d
```

In the same folder, remotely with a 16Gb RAM execute: (This remote inferecerer should be executed previously)
```bash
docker compose -f remote-docker-compose.yaml up
```



[FOBO]: https://github.com/izarte/FoBo
[mobile-net-for-fobo]: https://github.com/izarte/object-tracker-mobile-net-fobov2
[Gymnasium]: https://github.com/Farama-Foundation/Gymnasium
[PyBullet]: https://github.com/bulletphysics/bullet3
[train_fobo.py]: https://github.com/izarte/FoBov2/blob/main/train_fobo/train_utils/train_fobo.py
[docker-compose-train.yaml]: https://github.com/izarte/FoBov2/blob/main/train_fobo/docker/docker-compose-train.yaml
[Optuna]: https://github.com/optuna/optuna
[docker-compose-hyperparameters.yaml]: https://github.com/izarte/FoBov2/blob/main/train_fobo/docker/docker-compose-hyperparameters.yaml
[docker-compose-eval.yaml]: https://github.com/izarte/FoBov2/blob/main/train_fobo/docker/docker-compose-eval.yaml
[raspberry-installation]: https://github.com/izarte/FoBov2/blob/main/raspberry/README.md
[src]: https://github.com/izarte/FoBov2/tree/main/raspberry/src
[circuit]: https://github.com/izarte/FoBov2/blob/main/gallery/circuit.jpg
[robot]: https://github.com/izarte/FoBov2/blob/main/gallery/robot.jpg
