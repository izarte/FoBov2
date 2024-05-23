#!/usr/bin/expect -f

# Set variables
set source_folder "raspberry"
set destination_user "ubuntu"
set destination_host "192.168.1.254"
set destination_folder "/home/ubuntu/src"
set password $env(RASP_PASSWORD) ;

# Run rsync command with password provided automatically
spawn rsync -avz --force --progress $source_folder $destination_user@$destination_host:$destination_folder

# Expect password prompt
expect "password:"

# Send password
send "$password\r"

# Wait for rsync to complete
expect eof
