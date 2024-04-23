#!/bin/bash


# Run rsync command with password provided automatically

# Use scp with agent forwarding
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/docker/docker-compose.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/train_utils izarate@jackson.rovit.ua.es:/home/izarate/
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/fobo2_env izarate@jackson.rovit.ua.es:/home/izarate/