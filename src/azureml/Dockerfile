# Training/evaluation image for Azure ML jobs.
# Starts from an Azure ML curated base (Ubuntu + conda) and materializes the
# conda.yml environment. The project source is supplied at run time as the job
# `code` upload, so this image only provides the dependencies.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

COPY conda.yml /tmp/conda.yml

# Create the env from conda.yml and make it the default for the job process.
RUN conda env create -f /tmp/conda.yml -n project-env && conda clean -a -y
ENV PATH=/opt/conda/envs/project-env/bin:$PATH
ENV CONDA_DEFAULT_ENV=project-env
