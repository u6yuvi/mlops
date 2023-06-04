[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/tWFZppNq)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11257573&assignment_repo_type=AssignmentRepo)
# emlov3-session-02

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)


# PyTorch Docker Assignment

Welcome to the PyTorch Docker Assignment. This assignment is designed to help you understand and work with Docker and PyTorch.

## Assignment Overview

In this assignment, you will:

1. Create a Dockerfile for a PyTorch (CPU version) environment.
2. Keep the size of your Docker image under 1GB (uncompressed).
3. Train any model on the MNIST dataset inside the Docker container.
4. Save the trained model checkpoint to the host operating system.
5. Add an option to resume model training from a checkpoint.

## Starter Code

The provided starter code in train.py provides a basic structure for loading data, defining a model, and running training and testing loops. You will need to complete the code at locations marked by TODO: comments.

## Submission

When you have completed the assignment, push your code to your Github repository. The Github Actions workflow will automatically build your Docker image, run your training script, and check if the assignment requirements have been met. Check the Github Actions tab for the results of these checks. Make sure that all checks are passing before you submit the assignment.

## Getting Started 

### Build Docker Image
```
docker build -f Dockerfile -t mnist .
```

### Run Model training and testing from outside the container
```
docker run --name mnist_container --rm -v$(pwd):/workspace mnist python /workspace/train.py
```

### Run Model training from inside the container
```
docker run -it --name mnist_container mnist bash
python workspace/train.py
```

## Pushing Docker to Docker Hub
```
docker tag mnist u6yuvi/pytorch-cpu
docker push u6yuvi/pytorch-cpu
```

[Docker Hub Image Link](https://hub.docker.com/repository/docker/u6yuvi/pytorch-cpu/general)

# Maintainers

1. [Utkarsh Vardhan](https://github.com/u6yuvi)