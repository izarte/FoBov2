#!/bin/bash


# Run rsync command with password provided automatically

# Use scp with agent forwarding
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/docker/docker-compose.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/docker/docker-compose-train.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose-train.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/docker/docker-compose-hyper.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose-hyper.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/train_fobo.py izarate@jackson.rovit.ua.es:/home/izarate/train_fobo.py
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/train_fobo_optuna.py izarate@jackson.rovit.ua.es:/home/izarate/train_fobo_optuna.py
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/FoBov2/fobo2_env izarate@jackson.rovit.ua.es:/home/izarate/