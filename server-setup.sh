wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update
apt-get install libcudnn8=8.2.4.15-1+cuda11.4
apt-get install libcudnn8-dev=8.2.4.15-1+cuda11.4



wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
bzip2 -dk CTU-13-Dataset.tar.bz2
tar -xvf CTU-13-Dataset.tar

git clone https://github.com/rbnfrhnr/anomaly-detection.git
mkdir lbnl
cp anomaly-detection/Dockerfile lbnl/

docker build lbnl/