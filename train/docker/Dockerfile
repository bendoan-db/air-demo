# Training image for `databricks air run --file train/train.yaml`.
#
# Build and push it with train/docker/build_and_push.sh, which tags it as the
# Unity Catalog artifact train.yaml's environment.unity_catalog_image names. The
# build context is train/, so COPY paths below are relative to that directory.
#
# Base: Databricks' AI Runtime image for AWS. It ships CUDA 13.0, NCCL with
# EFA/aws-ofi-nccl preconfigured, Python 3.12 in a uv-managed virtualenv at
# /opt/venv (on PATH), and OPENSSL_FORCE_FIPS_MODE=0 for AI Runtime's
# FIPS-enabled hosts, but no ML libraries — this file adds the training stack.
# The base is amd64-only, so build with --platform linux/amd64.
# https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images
FROM databricksruntime/air:dcs-base-aws-runtime-cu13

# Package index for every install below. build_and_push.sh sets it to your
# machine's pip index when there is one, for networks that block public PyPI
# and route installs through a mirror. Never put credentials in this URL:
# build args are recorded in the image history, and pushed versions are
# immutable.
ARG PYPI_INDEX_URL=https://pypi.org/simple

# The training stack from the requirements file the runner notebook installs
# too, so CLI and notebook runs train on the same versions. PyPI's torch 2.11.0
# Linux wheel is already the CUDA 13 build (it depends on cuda-toolkit 13.0 and
# the -cu13 NVIDIA libraries), matching the base image, so it needs no separate
# PyTorch index.
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --index-url "$PYPI_INDEX_URL" -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fail here, not on a GPU node, if the installed torch is not the CUDA 13 build.
RUN python -c "import torch; assert (torch.version.cuda or '').startswith('13.'), torch.version.cuda; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"

# No COPY of the training code: code_source uploads train/ with every run and
# exposes it as $CODE_SOURCE_PATH, so code changes need no rebuild. AI Runtime
# ignores WORKDIR, which is why train.yaml's command uses an absolute path.
# Never bake credentials into this image: Artifact Registry versions are
# immutable and cannot be deleted.
