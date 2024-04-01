#!/bin/bash

python3 urdf/xacro.py -o urdf/fobo2.urdf urdf/fobo2.xacro
python3 urdf/xacro.py -o urdf/room.urdf urdf/room.xacro
python3 pybullet_test.py