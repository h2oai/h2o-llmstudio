#!/bin/bash -e


# Install core packages
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt update
sudo apt -y install curl
sudo apt -y install make

# Verify make installation
ls /usr/bin/make

# System installs (Python 3.10)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt -y install python3.10
sudo apt-get -y install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10


#add GPU support
set -eo pipefail
set -x

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey \
| sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get -y update
sudo apt-get install -y nvidia-container-runtime
rm cuda-repo-ubuntu2004-*.deb


# Clone h2o-llmstudio
git clone https://github.com/h2oai/h2o-llmstudio.git
cd h2o-llmstudio
git checkout "$VERSION" 


# Create virtual environment (pipenv)
make setup

# Running application as a service in systemd
cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .

cd /etc/systemd/system
printf """
[Unit]
Description=LLM Studio Service
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/h2o-llmstudio
ExecStart=/usr/bin/make llmstudio
Restart=always
[Install]
WantedBy=multi-user.target
""" >> llm_studio.service


sudo systemctl daemon-reload
sudo systemctl enable llm_studio.service
sudo systemctl start llm_studio.service

#Install nginx

sudo apt update
sudo apt install -y nginx


#configure nginx for port forwarding

cd /etc/nginx/conf.d
sudo chown -R ubuntu:ubuntu .
cd $HOME
printf """
server {
    listen 80;
    listen [::]:80;
    server_name <|_SUBST_PUBLIC_IP|>;  # Change this to your domain name

    location / {  # Change this if you'd like to server your Gradio app on a different path
        proxy_pass http://0.0.0.0:10101/; # Change this if your Gradio app will be running on a different port
        proxy_redirect off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
    }
}
""" > temp.conf

printf """
ip=\$(dig +short myip.opendns.com @resolver1.opendns.com)
sed \"s/<|_SUBST_PUBLIC_IP|>;/\$ip;/g\" /home/ubuntu/temp.conf  > /etc/nginx/conf.d/llm.conf
""" > run_nginx.sh

sudo chmod u+x run_nginx.sh

cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .
printf """
[Unit]
Description=LLM Nginx Server
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=bash /home/ubuntu/run_nginx.sh
Restart=always
[Install]
WantedBy=multi-user.target
""" > llm_nginx.service

sudo systemctl daemon-reload
sudo systemctl enable llm_nginx.service
sudo systemctl restart llm_studio.service
sudo systemctl restart nginx



