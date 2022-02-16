wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
bzip2 -dk CTU-13-Dataset.tar.bz2
tar -xvf CTU-13-Dataset.tar

git clone https://github.com/rbnfrhnr/anomaly-detection.git
mkdir lbnl
cp anomaly-detection/Dockerfile lbnl/

docker build lbnl/