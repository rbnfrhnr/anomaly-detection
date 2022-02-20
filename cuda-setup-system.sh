# Install Tesla T4 Driver 470.103.01
wget https://us.download.nvidia.com/tesla/470.103.01/NVIDIA-Linux-x86_64-470.103.01.run
chmod +x NVIDIA-Linux-x86_64-470.103.01.run
./NVIDIA-Linux-x86_64-470.103.01.run
# install cuda 11.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
apt-get update
apt-get -y install cuda