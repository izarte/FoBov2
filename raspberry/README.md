# Development

To develop this project I have use rsync to develop software in local system and send it to raspberry.

To do so, except (and rsync) package is needed, also RASP_PASSWORD should be the password ob raspbbery loaded as enviorment variable

# Instalation requirements

first plug both cameras and start

run lsusb to identify camera idVendor:idProduct

I my case I identify my device as following one
Bus 001 Device 003: ID 0c45:636d Microdia PC-LM1E
in this case idVendor=i0c45 and idProdcut=636d

sudo vim /etc/udev/rules.d/99-pi-camera.rules
ACTION=="add", ATTR{idVendor}=="0c45", ATTR{idProduct}=="636d", RUN="/bin/sh -c 'echo 0 >/sys/\$devpath/authorized'"

====OPTION 1======
check /sys/bus/usb/devices/ for list devices
got through each one reading idVendor file to identify which one is

cat /sys/bus/usb/devices/1-1.2/idVendor
0c45

add to crontab (crontab -e)
@reboot echo 1 | sudo tee /sys/bus/usb/devices/1-1.2/authorized

=====OPTION 2======
add this code to file $HOME/.utils/authorize_device.sh (make sure to create .utils folder) and give execution permissions (chmod +x $HOME/.utils/authorize_device.sh)

#!/bin/bash

# Target vendor ID
target_id="0c45"

# Base path for USB devices
base_path="/sys/bus/usb/devices"

# Iterate over all folders in the base path
for device_path in "$base_path"/*; do
    # Check if the idVendor file exists
    if [ -f "$device_path/idVendor" ]; then
        # Read the vendor ID from the file
        vendor_id=$(cat "$device_path/idVendor")

        # Check if the vendor ID matches the target ID
        if [ "$vendor_id" == "$target_id" ]; then
            # Print the matching device path
            #echo "Match found: $device_path"

            # Write 1 to the 'authorized' file of the device
            echo 1 | sudo tee "$device_path/authorized"
        fi
    fi
done

and add to crontab (crontab -e)
@reboot bash $HOME/.utils/authorize_device.sh



Give user access to google coral
sudo usermod -aG plugdev $USER