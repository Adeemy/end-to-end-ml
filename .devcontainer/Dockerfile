FROM python:3.10-bullseye

# Set environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install appropriate conda based on machine architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh; \
    else \
        echo "Unsupported machine architecture: $ARCH" && exit 1; \
    fi && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Activate conda
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc

# Set the default command to run when starting the container
CMD ["conda", "run", "--no-capture-output", "-n", "/bin/bash"]
