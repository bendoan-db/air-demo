# DAIS AI Runtime Demo

This project contains a Databricks AI Runtime demo for fine-tuning a small language model on credit-card fraud transactions, registering it as a custom LLM, deploying it to Mosaic AI Model Serving, and load testing the deployed endpoint.

The demo uses the IBM TabFormer credit-card dataset and prepares a supervised fine-tuning table where each row contains a transaction prompt and target assistant response.

## Run the Demo

The demo fine-tunes `Qwen/Qwen3.5-4B` two ways from the same code and configuration: first by submitting training from your terminal with the AI Runtime CLI, then interactively in the runner notebook, which also registers, deploys, and queries the model. Both paths read `train/train.yaml` and log to the same MLflow experiment.

### Prerequisites

Workspace:

- Databricks workspace with Unity Catalog enabled.
- A Unity Catalog catalog that already exists.
- Permission to create schemas, volumes, tables, registered models, and serving endpoints in the target catalog/schema.
- Databricks serverless compute for ingestion and load testing, on environment version 5 or above (the notebooks install packages with `%uv pip`).
- Databricks Serverless GPU with AI Runtime for training.
- Model Serving access with GPU workloads enabled for custom LLM serving.
- Local Databricks CLI authentication if running notebooks or scripts from this repository with Databricks Connect.

For the CLI path (Part 1):

- Databricks CLI v1.19.0 or above, authenticated with a workspace OAuth profile from `databricks auth login` (the Docker credential helper doesn't support personal access tokens). If several profiles in `~/.databrickscfg` point at the same workspace, only one of them may have a `workspace_id` line; see [If the build or push fails](#if-the-build-or-push-fails).
- Docker installed, running, and on your `PATH` on the machine you build from, for example Docker Desktop (whose CLI lives in `~/.docker/bin`). The base image is amd64-only, so Docker must be able to build `linux/amd64` images; on Apple Silicon that means amd64 emulation, which Docker Desktop provides out of the box. The base image is public, so no Docker Hub login is needed, and the push script sets up Artifact Registry authentication itself. If your machine blocks public PyPI and routes pip through a mirror (as Databricks laptops do), the script builds with that mirror; see Part 1 step 3. Leave room on disk: the base image alone is about 2.7 GB compressed, and the CUDA 13 torch wheels add several GB more.
- The **AI Runtime Beta Features** and **Databricks Artifact Registry** previews, enabled by a workspace admin.
- An existing catalog and schema for the image (Artifact Registry doesn't create them), with `USE CATALOG`, `USE SCHEMA`, and `CREATE VOLUME` on them — plus `WRITE VOLUME` to push later versions.

### Before you start

1. With the [prerequisites](#prerequisites) in place, update the [configuration](#configuration) — at minimum `catalog` and `schema` in all three YAML files, and `mlflow_experiment_directory` plus `environment.unity_catalog_image` in `train/train.yaml`.
2. Prepare the data. In the workspace, run `setup/01_load_tabformer_dataset.py` on serverless compute (environment version 5 or above). Both training paths read the Parquet shard export it writes; it overwrites its tables on every run, so you only need to rerun it when the data or prompt format changes.
3. Optional: run `setup/02_load_model_weights_to_volume.py` to mirror the base model into a Unity Catalog volume, then set `train/train.yaml`'s `model_name` to the printed `/Volumes/...` path. Otherwise every training run downloads the weights from the Hugging Face Hub.

### Part 1: Train with the AI Runtime CLI

Submit `train/train.py` to serverless GPUs from your terminal with the Databricks CLI's [`air` command group](https://docs.databricks.com/aws/en/dev-tools/cli/reference/air-commands), without opening a notebook. The workload runs in a custom container image and logs to MLflow.

1. Install or update the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) to v1.19.0 or above (the minimum for running custom images) and confirm the `air` commands are available:

   ```bash
   databricks air --help
   ```

   These commands replace the deprecated Python-based `air` CLI from the `databricks-air` package, which uses different command names and flags. If you installed it earlier, remove it with `uv tool uninstall databricks-air`.

   Optionally, install the `databricks-ai-runtime` skill so coding agents (Claude Code, Cursor, Codex) can write workload YAML and submit and monitor runs: `databricks aitools install --skills-only --skills databricks-ai-runtime --experimental`.

2. Authenticate. The CLI uses profiles from `~/.databrickscfg`; add `-p <profile>` to any command to pick one:

   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   ```

3. Build the training image and push it to Unity Catalog Artifact Registry (once, and again only when `train/requirements.txt` changes). `train/docker/Dockerfile` starts from Databricks' AI Runtime base image `databricksruntime/air:dcs-base-aws-runtime-cu13` (CUDA 13.0, NCCL/EFA, Python 3.12, and the FIPS setting preconfigured) and adds the same `train/requirements.txt` the notebook installs. [Custom images](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) and [Artifact Registry](https://docs.databricks.com/aws/en/artifact-registry/get-started) are in Beta, so check the [CLI-path prerequisites](#prerequisites) first — Docker, the previews, and the image's catalog, schema, and grants.

   ```bash
   train/docker/build_and_push.sh --profile <profile>
   ```

   The script takes its destination from `environment.unity_catalog_image` in `train/train.yaml` (`ssa_team_sandbox_classic_catalog.air.air_demo_training:v1`), so change the image name or tag there, not in the script. It builds for `linux/amd64` (the base image is amd64-only, which also makes Apple Silicon builds work) and installs Python packages from your machine's pip index — `--index-url`, else `$PIP_INDEX_URL`, else the `index-url` in your pip config, else public PyPI — because the build can't see your local pip settings and some networks block public PyPI. It then runs `databricks auth docker configure` to set up Docker authentication for the workspace registry, tags and pushes the image as `<registry-host>/<catalog>.<schema>.<artifact>:<tag>`, and lists the artifact's versions. The build fails early if torch isn't the CUDA 13 build. The training code is not baked into the image; it ships with every run. Pushing an existing tag again moves it to the new version; bump the tag instead to keep earlier runs reproducible.

   The first build downloads the 2.7 GB base image and about 3 GB of Python packages (mostly torch and its CUDA libraries), and produces an image of about 5.6 GB (the limit is 20 GB). Later builds reuse Docker's cache until `train/requirements.txt` changes. The first push uploads all 5.6 GB (about 20 minutes on a typical connection); later pushes skip layers the registry already has.

   #### If the build or push fails

   - **`docker is installed in ~/.docker/bin ... but that directory is not on PATH`**: Docker Desktop adds `~/.docker/bin` to `PATH` in `~/.zprofile`, which only new login shells read. Open a new terminal window, or run `export PATH="$PATH:$HOME/.docker/bin"`. If the error persists in a new window (some IDE terminals aren't login shells), copy that line into `~/.zshrc`.
   - **`the Docker daemon is not running`**: start Docker Desktop.
   - **`Connection refused` while installing packages, or `public PyPI is blocked on this machine`**: your machine blocks public package indexes (Databricks laptops do this with a managed `/etc/hosts`), and the build container can't see your local pip settings. The script passes in the `index-url` from your pip config automatically; if you have none, pass your mirror with `--index-url <url>` or `PIP_INDEX_URL`. The mirror URL must not contain credentials.
   - **`multiple Databricks profiles match workspace ID ...`**: the Docker credential helper finds the CLI profile for a registry by workspace ID, so only one profile per workspace may set `workspace_id` in `~/.databrickscfg`. Keep it on the profile you pass with `--profile`, and delete the `workspace_id` line from the other profiles the error lists (or remove those profiles).
   - **`Databricks Artifact Registry is not enabled for this workspace`, or `403 Forbidden` from the push**: a workspace admin must turn on the **Databricks Artifact Registry** preview on the workspace's Previews page (the script checks for this before building). If the preview is on and the push still returns 403, check your grants on the image's schema: `USE CATALOG`, `USE SCHEMA`, and `CREATE VOLUME` for the first push, or `WRITE VOLUME` for later versions.
   - **`Internal Server Error` at the end of the push, after the layers have uploaded**: the registry rejects the multi-entry image index that Docker Desktop builds by default (the image plus provenance attestations), and no version gets registered. The script builds without attestations (`--provenance=false --sbom=false`) and pushes a single-platform manifest, so this only affects images built another way: push one of those with `docker push --platform linux/amd64 <registry-host>/<catalog>.<schema>.<artifact>:<tag>`. The layers already uploaded are reused.
   - **`refresh token is invalid`, or other authentication errors**: run `databricks auth login --profile <profile>` again.

   Once a push-side problem is fixed, rerun with `--skip-build` to push the image you already built instead of rebuilding it.

4. Validate the workload, then submit it:

   ```bash
   databricks air run --file train/train.yaml --dry-run
   databricks air run --file train/train.yaml --watch
   ```

   `--dry-run` checks the YAML locally without submitting. `--watch` streams node 0's logs until the run finishes; the last lines print the MLflow run ID and the adapter directory on the checkpoint volume. `train.yaml` requests one `GPU_8xH100` node; for a cheaper first run, add `--override compute.accelerator_type=GPU_1xA10 --override compute.num_accelerators=1`.

5. Monitor and manage runs with the Job Run ID that `databricks air run` prints:

   ```bash
   databricks air list                  # your active runs (--all-status includes finished ones)
   databricks air get <job-run-id>      # status, configuration, timing, and links to the job and MLflow run
   databricks air logs <job-run-id>     # stream logs (node 0 by default; --download-to <dir> saves them)
   databricks air cancel <job-run-id>   # stop a run
   ```

The CLI path runs training only. To register and deploy the adapter it produced, use the runner notebook (Part 2, step 3). For overrides, scaling, and other CLI details, see [AI Runtime CLI notes](#ai-runtime-cli-notes).

### Part 2: Run the notebook runner

`train/runner.py` runs the same training interactively with the `@distributed` decorator, then merges, registers, deploys, and queries the model.

1. Open `train/runner.py` in the workspace (for example, from a Git folder) and attach it to **Serverless GPU** with the **AI v6** environment. Use `1xH100` or `1xA10` for the first run, or `8xH100` to show multi-GPU scaling.

2. Run the notebook from the top:

   1. **Install and configure**: installs `train/requirements.txt` with `%uv pip`, restarts Python, loads `train/train.yaml`, and creates the schema and checkpoint volume if needed.
   2. **Fine-tune**: the training cell runs on one GPU with `@distributed(gpus=1, gpu_type="h100")`. To demonstrate scaling, change only `gpus` (for example to `8`) and rerun the cell — the training code stays the same.
   3. **Merge, install the serving stack, register**: three cells, separated by a `%restart_python`, that merge the adapter, install vLLM, and register the model to Unity Catalog (skipped when `register_model: false`).
   4. **Deploy**: creates or updates the Model Serving endpoint (skipped when `deploy_endpoint: false`) and waits until it's ready.
   5. **Query**: prints a sample request payload and sends a test request to the endpoint.

3. To register an adapter trained with the CLI instead, run the cells through the configuration step, skip the training cell, and define the two values it would have set, using the ones the CLI run printed:

   ```python
   TRAINED_ADAPTER_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/8gpu"  # "Trained adapter output dir" from the CLI log
   TRAINING_RUN_ID = "<Training MLflow run ID from the CLI log>"
   ```

   Then continue from the merge cell.

4. Load test the endpoint. Once the endpoint is ready, run `load_test/load_test_serving_endpoint.py` on serverless compute (environment version 5 or above). It samples prompts from the SFT table, runs a smoke test, generates paced high-QPS traffic from Spark tasks, and appends the results to a Delta table.

## Project Layout

| Path | Purpose |
| --- | --- |
| `setup/01_load_tabformer_dataset.py` | Databricks notebook that downloads TabFormer, cleans transaction data, and overwrites Delta tables. |
| `setup/02_load_model_weights_to_volume.py` | Databricks notebook that mirrors the base model weights from the Hugging Face Hub into a Unity Catalog volume. |
| `setup/setup.yaml` | Ingestion configuration: catalog, schema, table names, staging volume, source URL, SFT shard count, and the base-model mirror settings. |
| `train/runner.py` | Databricks notebook for AIR fine-tuning with Hugging Face TRL, MLflow registration, and Model Serving deployment. |
| `train/train.py` | Standalone training module: imported by the notebook's `@distributed` cell and runnable directly via the AI Runtime CLI. |
| `train/train.yaml` | AI Runtime CLI workload definition (`databricks air run --file train/train.yaml`) plus the training, registration, and serving configuration (`parameters.training_config` section). |
| `load_test/load_test_serving_endpoint.py` | Databricks notebook that simulates high-QPS traffic against the deployed serving endpoint. |
| `load_test/serving_load_test.yaml` | Load-test configuration. |
| `train/training_utils.py` | Shared notebook utilities for YAML config loading and Unity Catalog name handling. |
| `train/requirements.txt` | Training dependencies, installed by the runner notebook and baked into the training image. |
| `train/docker/Dockerfile` | Custom training image for CLI runs: Databricks' AI Runtime base image plus `train/requirements.txt`. |
| `train/docker/build_and_push.sh` | Builds the image and pushes it to Unity Catalog Artifact Registry under `train.yaml`'s `environment.unity_catalog_image`. |
| `databricks.yml` | Databricks bundle metadata used by the Databricks extension/CLI. |

## Configuration

Update these files before running the demo:

- `setup/setup.yaml`
  - `catalog` and `schema`
  - `table` and `sft_table`
  - `sft_volume` (volume for the Parquet export of the SFT table)
  - `staging_volume`
  - `source_url`
  - `model_name` (Hugging Face repo id of the base checkpoint; must match `train.yaml`'s `model_name`)
  - `model_volume` and `model_revision` (destination volume and Hub revision for the mirrored weights)

- `train/train.yaml`, in three sections:
  - **Required AI Runtime fields**: `experiment_name` (the MLflow experiment both the notebook and the CLI log to), `compute` (accelerator type and count), and `command`
  - **Optional AI Runtime fields**: `environment.unity_catalog_image` (the Artifact Registry image, as `<catalog>.<schema>.<artifact>:<tag>`; `train/docker/build_and_push.sh` pushes to this name), `code_source`, `mlflow_experiment_directory` (the `/Workspace/...` folder holding the experiment; remove it to fall back to your home folder), and `max_retries`
  - **Demo configuration** (`parameters.training_config`):
    - `catalog`, `schema`, `source_table`, and `sft_table`
    - `checkpoint_volume`
    - `uc_model_name`
    - `endpoint_name`
    - training parameters such as `max_steps`, `training_sample_fraction`, batch size, and learning rate
    - serving parameters such as `serving_workload_type`, `serving_workload_size`, and `serving_scale_to_zero`

- `load_test/serving_load_test.yaml`
  - `endpoint_name` (must match `train.yaml`)
  - `enable_thinking` (must match the training render)
  - `target_qps`
  - `duration_seconds`
  - load-generator worker and concurrency settings

## How It Works

1. **Ingest and prepare the dataset** (`setup/01_load_tabformer_dataset.py`):

   - Creates the configured schema and staging volume if they do not exist.
   - Downloads and extracts the IBM TabFormer transactions archive.
   - Standardizes transaction columns and data types.
   - Adds prompt-ready transaction fields and fraud labels.
   - Writes the cleaned transaction Delta table.
   - Writes the prepared SFT Delta table with prompt, response, and shard columns.
   - Exports the SFT records to a Unity Catalog volume as Parquet files partitioned by `shard_id`, per the [AI Runtime data-loading guidance](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-unity-catalog-volumes).
   - Overwrites target tables on each run.

2. **Mirror the base model weights** (optional, `setup/02_load_model_weights_to_volume.py`):

   - Creates the configured model volume if it does not exist.
   - Resolves `model_revision` to a commit SHA and downloads every file of that snapshot except the `model_ignore_patterns` matches.
   - Copies each file to the volume one at a time, deleting the local copy in between, so local disk use stays bounded and volume writes stay sequential.
   - Skips files already mirrored at the size the Hub reports, unless `force_model_download` is set.
   - Writes a provenance JSON beside the weights directory and verifies the snapshot (config, tokenizer, and every safetensors shard named in the index).

3. **Fine-tune** (`train/train.py`, launched by the CLI or the runner notebook):

   - Reads the rank-sharded SFT Parquet files from the Unity Catalog volume with Hugging Face `datasets` (no Spark on the GPU workers); each rank loads only its own `shard_id` directories.
   - Fine-tunes `Qwen/Qwen3.5-4B` (text backbone only) with TRL supervised fine-tuning and PEFT LoRA, computing loss on the assistant response only, with thinking suppressed via `enable_thinking=False`.
   - Runs one process per GPU: `torchrun` under the CLI, the `@distributed` decorator in the notebook.
   - Saves rank-0 adapter artifacts to a Unity Catalog volume.
   - Logs training metrics to MLflow.

4. **Register the custom LLM** (runner notebook), split into three cells because the training and serving environments cannot coexist in one Python session:

   - **Merge**: loads the saved adapter, merges it into the base model, and writes merged Hugging Face weights to `/local_disk0`.
   - **Install the serving stack**: two `%uv pip` passes — `vllm==0.24.0`, `transformers==5.13.0`, `mlflow==3.14.0`, `flashinfer-cubin`, then `opencv-python-headless==4.12.0.88` on top (this deliberately breaks vLLM's `opencv>=4.13` floor, which `%uv pip` does not check against installed packages; anything `>=4.13` fails the OpenSSL FIPS self-test on Model Serving pods) — followed by `%restart_python`.
   - **Register**: configures a vLLM OpenAI-compatible server entrypoint for `llm/v1/chat` with `--language-model-only`, and registers the MLflow model to Unity Catalog with `env_pack="databricks_model_serving"`, which captures the environment installed above.

   Note for anyone reusing this pin set: `opencv-python-headless<4.13` carries a known RCE CVE, which is why the managed Foundation Model path does not ship this combination.

5. **Deploy the serving endpoint** (runner notebook): if `deploy_endpoint: true` in `train/train.yaml`'s `training_config`, the notebook creates or updates the configured Model Serving endpoint and routes 100% of traffic to the registered model version.

6. **Load test the endpoint** (`load_test/load_test_serving_endpoint.py`): samples prompts from the SFT Delta table, runs a smoke test, generates asynchronous HTTP traffic from Spark tasks, and records achieved throughput, status counts, latency samples, and summary metrics to a Delta table.

## AI Runtime CLI Notes

- **Overrides**: change config values per run without editing the file, one `--override` per field:

  ```bash
  databricks air run --file train/train.yaml \
    --override parameters.training_config.max_steps=50 \
    --override parameters.training_config.training_sample_fraction=0.01 \
    --watch
  ```

- **Paths**: `code_source.root_path` resolves against the YAML file's directory, so the commands work from the repo root or from `train/`. The code snapshot is built from `git ls-files` in `train/`: tracked files plus any untracked files that `.gitignore` doesn't exclude, so stray local files in `train/` ship with the run. Run `databricks air run -h config` (or `-h config.<field>`) for schema help on any workload field.

- **Retries**: `train.yaml` sets `max_retries: 0`, so a failed run stops instead of retrying. The CLI default is 3, and each retry is a fresh workload and a new MLflow run in the same experiment; raise it for long unattended runs, for example with `--override max_retries=2`.

- **MLflow**: runs land in the same MLflow experiment as notebook runs — AIR creates `experiment_name` inside `mlflow_experiment_directory` (your home folder if that is unset), and the notebook resolves the same two fields through `training_utils.resolve_experiment_path()`. Two markers distinguish the launch path: the run name carries an `-air-cli` suffix and the run is tagged `submitted_via: air-cli` (notebook runs are tagged `submitted_via: notebook`). Filter with `tags.submitted_via = 'air-cli'` in the MLflow UI. AI Runtime captures GPU, CPU, and memory system metrics for CLI runs automatically, on the MLflow run's System metrics tab.

- **Scaling**: `train.py` resolves rank and world size from `torchrun`. Requesting more accelerators than one node holds spans several nodes, and AI Runtime runs `command` once per node, so that also means replacing `--standalone` with the rendezvous arguments described in `train.yaml`'s `command` comment.

- **Without the custom image**: if the Beta previews aren't available, replace `train.yaml`'s `environment` block with `version: databricks_ai_v6` plus a `dependencies:` list of `train/requirements.txt`'s packages to run on the managed AI environment.

- **Scheduling**: to schedule the training workload or chain it after ingestion as a job, `databricks air convert-to-dabs train/train.yaml` generates a Declarative Automation Bundle starting point (`databricks.yml` plus `generated_artifacts/`, written next to the YAML unless you pass `--output-dir`). See [Schedule GPU workloads and compose tasks](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/productionizing-training-workloads).

## Local Development

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install local dependencies:

```bash
.venv/bin/python -m pip install -r train/requirements.txt
```

The ingestion notebook can run with Databricks Connect when authentication is configured:

```bash
databricks auth profiles
.venv/bin/python setup/01_load_tabformer_dataset.py
```

The training notebook is intended to run on Databricks Serverless GPU because it depends on AI Runtime, GPU hardware, and the `serverless_gpu` distributed runtime.

## References

- Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
- Databricks CLI with AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/
- `air` command group: https://docs.databricks.com/aws/en/dev-tools/cli/reference/air-commands
- Custom Docker images with AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images
- Get started with Artifact Registry: https://docs.databricks.com/aws/en/artifact-registry/get-started
- Databricks custom LLM serving: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms
- IBM TabFormer dataset: https://github.com/IBM/TabFormer
