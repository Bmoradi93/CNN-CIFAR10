# This is a simple instruction to run my coding question #4: Behnam Moradi

#### I assume you are using ubuntu 20.04 as your OS

#### Installing ther dependencies and python env:
  Please open a terminal and execute the following commands to install your dependencies:

sudo apt-get update 
sudo apt-get install -y build-essential cmake git unzip pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libhdf5-serial-dev graphviz
sudo apt-get install -y libopenblas-dev libatlas-base-dev gfortran
sudo apt-get install -y python-tk python3-tk python-imaging-tk
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo pip3 install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip get-pip.py
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc


#### Please create and enter the python env using the following commands:

mkvirtualenv assignment4 -p python3

workon assignment_4

pip3 install numpy
pip3 install opencv-contrib-python
pip3 install scipy matplotlib pillow
pip3 install imutils h5py requests progressbar2
pip3 install scikit-learn scikit-image
pip3 install tensorflow==2.2.0
pip3 install keras
pip3 install pandas
pip3 install pyyaml

#### You are ready to run my code.