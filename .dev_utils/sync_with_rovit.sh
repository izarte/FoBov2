#!/bin/bash


# Run rsync command with password provided automatically

# Use scp with agent forwarding
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/fobov2/train_fobo/docker/docker-compose-rovit.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose.yaml
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/fobov2/train_fobo/docker/docker-compose-hyperparameters.yaml izarate@jackson.rovit.ua.es:/home/izarate/docker-compose-hyperparameters.yaml
rsync -avz --force --progress --exclude='trained_models/' -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/fobov2/train_fobo/train_utils izarate@jackson.rovit.ua.es:/home/izarate/
rsync -avz --force --progress -e "ssh -i /home/inigo/.ssh/private_keys/rovit_np -p 8080" /home/inigo/fobov2/train_fobo/fobo2_env izarate@jackson.rovit.ua.es:/home/izarate/