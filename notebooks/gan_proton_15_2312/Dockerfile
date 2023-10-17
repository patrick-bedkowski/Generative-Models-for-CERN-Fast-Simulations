# Use a base image with the desired version of TensorFlow already installed.
FROM tensorflow/tensorflow:latest-gpu

# Upgrade pip, urllib3, and chardet
RUN /usr/bin/python3 -m pip install --upgrade pip urllib3 chardet


# Install pandas and numpy.
RUN pip install wandb imageio pandas==1.5.1 numpy==1.23.4 matplotlib scikit-learn ipython jupyter