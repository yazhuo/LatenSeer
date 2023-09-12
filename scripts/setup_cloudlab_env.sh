#!/bin/bash
# This script installs all libraries to meet DSB's requirements
# This script was tested on small-lan profile/Ubuntu 20.04 on cloudlab

# Install Docker
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.2.3/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
#Optional: sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install pip3, asyncio, and aiohttp
sudo apt-get -y install python3-pip
pip3 install asyncio
pip3 install aiohttp

# Other libraries
sudo apt-get -y install libssl-dev
sudo apt-get -y install libz-dev
sudo apt-get -y install luarocks
sudo luarocks install luasocket

# Manage Docker as a non-root user
sudo usermod -a -G docker $USER
newgrp docker