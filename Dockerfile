# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04

# The linux user which will be created
ARG user=user

# Set the timezone environmental variable
ENV TZ=Europe/London

# Update the apt sources
RUN apt update

# Install pip so that we can install PyTorch
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3.8 python3-pip

# Install PyTorch
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Unminimize Ubunutu, and install a bunch of necessary/helpful packages
RUN yes | unminimize
RUN DEBIAN_FRONTEND=noninteractive apt install -y sudo ubuntu-server openssh-server python-is-python3 git python3-venv

# Create a new user and allow them passwordless sudo
RUN useradd --create-home --groups sudo --shell /bin/bash ${user} \
    && echo '%sudo	ALL = (ALL) NOPASSWD: ALL' > /etc/sudoers.d/passwordless-sudo

# Switch to this new user
USER ${user}
WORKDIR /home/${user}

# Do all the things which require secrets: set up git, login to Weights &
# Biases and clone the repo
RUN --mount=type=secret,id=my_env,mode=0444 /bin/bash -c 'source /run/secrets/my_env \
    && git config --global user.name "${GIT_NAME}" \
    && git config --global user.email "${GIT_EMAIL}" \
    && git clone https://$GITHUB_USER:$GITHUB_PAT@github.com/SamAdamDay/berkeleydeeprl_homework_fall2022.git Homeworks \
    && mkdir -p .ssh \
    && echo ${SSH_PUBKEY} > .ssh/authorized_keys'

# Add ~/.local/bin to the path
ENV PATH=/home/${user}/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Move to the repo directory
WORKDIR Homeworks

# Install all the required packages
RUN pip install --user --upgrade pip \
    && pip install --user wheel \
    && pip install --user -r requirements.txt \
    && pip install --user nvitop

# Go back to the root
USER root
WORKDIR /

# Expose the default SSH port (inside the container)
EXPOSE 22
