#!/bin/bash


# Run rsync command with password provided automatically

# Use scp with agent forwarding
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/paperspace" /home/inigo/FoBov2/docker/docker-compose-paperspace.yaml paperspace@184.105.3.141:/home/paperspace/docker-compose.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/paperspace" /home/inigo/FoBov2/docker/docker-compose-hyperparameters.yaml paperspace@184.105.3.141:/home/paperspace/docker-compose-hyperparameters.yaml
rsync -avz --force --progress --exclude='trained_models/' -e "ssh -i /home/inigo/.ssh/private_keys/paperspace" /home/inigo/FoBov2/train_utils paperspace@184.105.3.141:/home/paperspace/
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/paperspace" /home/inigo/FoBov2/fobo2_env paperspace@184.105.3.141:/home/paperspace/