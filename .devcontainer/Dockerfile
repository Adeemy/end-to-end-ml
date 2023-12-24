FROM python:3.10

# Install and update packages
RUN apt update && \
    apt install -y sudo

# Allow multiple commands to be run in a single line 
# to activate venv when container starts and installs 
# required libraries.
CMD ["/bin/bash", "-c"]