Deep Reinforcement Learning Homeworks
=====================================

My solutions for the assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

I develop and run things using [vast.ai](https://vast.ai/). For this I've built a handy Docker image (see below).


Using Docker
------------

A docker file is available which allows for iterative development and running experiments. To build a new image and use it, follow the proceeding steps.

1. Create GitHub personal access token. Ideally use a fine-grained one which has access only to the contents of this repository.

2. Create a file named `.env` with the following contents

```bash
GITHUB_USER=
GITHUB_PAT=
GIT_NAME=""
GIT_EMAIL=""
SSH_PUBKEY=""
```

3. Fill in the details with your GitHub username, your GitHub PAT, your name as you'd like it to appear in git commit messages, the email you'd like to use for git commits and the SSH public key you'd like to use to access the container.

4. Build the image using the following command:

```
docker build -t DOCKER_REPO:DOCKER_TAG --secret id=my_env,src=.env .
```

replacing `DOCKER_REPO` and `DOCKER_TAG` with the appropriate details.

5. Push the image to the Docker Hub, ready for use.


Using with Vast.ai
------------------

1. Use the docker image you built and uploaded to the Hub.
2. Set the 'on-start script' to be [`vastai-startup-script`](/vastai-startup-script).