#!/usr/bin/env bash
# Build the AI Runtime training image and push it to Databricks Artifact Registry.
#
# The destination is the image train/train.yaml runs: its
# environment.unity_catalog_image, <catalog>.<schema>.<artifact>:<tag>. Reading it
# from there keeps the pushed image and the workload from drifting apart. The push
# follows https://docs.databricks.com/aws/en/artifact-registry/get-started:
# configure Docker authentication for the workspace registry, tag the image under
# the registry host, and push it.
#
# Usage: train/docker/build_and_push.sh [--profile PROFILE] [--index-url URL] [--skip-build]
#   --profile PROFILE  Databricks CLI profile for the target workspace (default:
#                      $DATABRICKS_CONFIG_PROFILE, else DEFAULT). It must be a
#                      workspace OAuth profile from `databricks auth login`; the
#                      Docker credential helper does not support PAT or M2M
#                      profiles. The first run saves the workspace_id into it.
#   --index-url URL    Python package index the image build installs from
#                      (default: $PIP_INDEX_URL, else the index-url in your pip
#                      config, else public PyPI). Needed where public PyPI is
#                      blocked and installs go through a mirror. Must not
#                      contain credentials: build args land in the image history.
#   --skip-build       Push the existing local image instead of rebuilding it.
#
# Requires Docker, Databricks CLI 1.18.0+, the Databricks Artifact
# Registry preview, an existing catalog and schema, and on that schema USE CATALOG
# + USE SCHEMA plus CREATE VOLUME (first push of the artifact) or WRITE VOLUME (a
# new version or a moved tag). Running the image needs READ VOLUME.
set -euo pipefail

# This script lives in train/docker/; train/ is both the build context and
# where the workload YAML lives.
train_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workload_yaml="$train_dir/train.yaml"
profile="${DATABRICKS_CONFIG_PROFILE:-DEFAULT}"
index_url=""
skip_build=false

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p | --profile)
      [[ $# -ge 2 ]] || die "--profile needs a value"
      profile="$2"
      shift 2
      ;;
    --index-url)
      [[ $# -ge 2 ]] || die "--index-url needs a value"
      index_url="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    -h | --help)
      # Print the header comment above as usage.
      awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

for tool in docker databricks; do
  command -v "$tool" >/dev/null && continue
  # Docker Desktop's per-user install links its CLI and credential helpers into
  # ~/.docker/bin and adds that to PATH only in ~/.zprofile, which non-login
  # shells and terminals opened before the install never read.
  if [[ "$tool" == docker && -x "$HOME/.docker/bin/docker" ]]; then
    die "docker is installed in ~/.docker/bin (Docker Desktop's per-user location), but that directory" \
      "is not on PATH. Open a new terminal window, or run: export PATH=\"\$PATH:\$HOME/.docker/bin\""
  fi
  die "$tool is not installed or not on PATH"
done
docker info >/dev/null 2>&1 || die "the Docker daemon is not running"

# environment.unity_catalog_image, the only key with that name in train.yaml.
image_ref="$(sed -n 's/^[[:space:]]*unity_catalog_image:[[:space:]]*//p' "$workload_yaml" |
  head -n 1 | sed -e 's/[[:space:]]*#.*$//' -e "s/^[\"']//" -e "s/[\"']\$//")"
[[ -n "$image_ref" ]] || die "no environment.unity_catalog_image in $workload_yaml"

# <catalog>.<schema>.<artifact>:<tag>, checked against Artifact Registry's rules
# up front so a bad name fails before a multi-GB build.
ref_re='^([^./:[:space:]]+)\.([^./:[:space:]]+)\.([^./:[:space:]]+):([^./:[:space:]][^/:[:space:]]*)$'
[[ "$image_ref" =~ $ref_re ]] ||
  die "unity_catalog_image must be <catalog>.<schema>.<artifact>:<tag>, got '$image_ref'"
catalog="${BASH_REMATCH[1]}"
schema="${BASH_REMATCH[2]}"
artifact="${BASH_REMATCH[3]}"
tag="${BASH_REMATCH[4]}"

artifact_re='^[a-z0-9]([a-z0-9_-]*[a-z0-9])?$'
if [[ ! "$artifact" =~ $artifact_re || "$artifact" == *___* || "$artifact" == *_-* ||
  "$artifact" == *-_* || ${#artifact} -gt 255 ]]; then
  die "artifact name '$artifact' breaks Artifact Registry's rules: lowercase letters, digits, _ and -;" \
    "starts and ends with a letter or digit; at most two consecutive underscores; no underscore next to a hyphen;" \
    "at most 255 characters"
fi
tag_re='^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$'
[[ "$tag" =~ $tag_re ]] || die "'$tag' is not a valid Docker tag"

local_image="$artifact:$tag"
artifact_name="$catalog.$schema.$artifact"

# Fail before a multi-GB build if Artifact Registry is off in this workspace: the
# push would otherwise end in a bare "403 Forbidden". Any other error from this
# read-only call (expired login, no READ METADATA) is left for the steps below.
if ! ar_check="$(databricks api get --profile "$profile" \
  "/api/2.1/unity-catalog/software-artifacts?parent=schemas/$catalog.$schema" 2>&1)"; then
  if grep -qi 'not enabled' <<<"$ar_check"; then
    die "Databricks Artifact Registry is not enabled for this workspace (profile $profile)." \
      "A workspace admin must turn on the Databricks Artifact Registry preview on the workspace's Previews page."
  fi
fi

if [[ "$skip_build" == false ]]; then
  # Package index for the build: --index-url, then $PIP_INDEX_URL, then the
  # index-url in the first user pip config found, else the Dockerfile's PyPI
  # default. The container cannot see host pip/uv config, so a machine that
  # routes installs through a mirror has to hand it in explicitly.
  index_source="--index-url"
  if [[ -z "$index_url" && -n "${PIP_INDEX_URL:-}" ]]; then
    index_url="$PIP_INDEX_URL" index_source="\$PIP_INDEX_URL"
  fi
  if [[ -z "$index_url" ]]; then
    for pip_conf in "$HOME/.config/pip/pip.conf" "$HOME/Library/Application Support/pip/pip.conf" "$HOME/.pip/pip.conf"; do
      [[ -f "$pip_conf" ]] || continue
      index_url="$(sed -n 's/^[[:space:]]*index[-_]url[[:space:]]*=[[:space:]]*//p' "$pip_conf" | head -n 1 | sed 's/[[:space:]]*$//')"
      if [[ -n "$index_url" ]]; then
        index_source="$pip_conf"
        break
      fi
    done
  fi
  build_args=()
  if [[ -n "$index_url" ]]; then
    [[ "$index_url" != *://*@* ]] ||
      die "the package index URL contains credentials, which would be baked into the image history;" \
        "use a mirror that needs no inline credentials"
    echo "==> Installing Python packages from $index_url (from $index_source)"
    build_args=(--build-arg "PYPI_INDEX_URL=$index_url")
  elif grep -Eq '^[[:space:]]*(127\.0\.0\.1|0\.0\.0\.0|::1)[[:space:]]+([^#]*[[:space:]])?pypi\.org([[:space:]]|$)' /etc/hosts 2>/dev/null; then
    # Docker Desktop resolves names through the host, so the build would fail
    # partway with "Connection refused".
    die "public PyPI is blocked on this machine (/etc/hosts maps pypi.org to loopback);" \
      "pass your PyPI mirror with --index-url URL or set PIP_INDEX_URL"
  fi

  echo "==> Building $local_image from train/docker/Dockerfile"
  # The base image is amd64-only, so pin the platform (this also makes Apple
  # Silicon builds work). No provenance/SBOM attestations: with them, Docker
  # Desktop stores the image as an OCI index (image + attestation manifest),
  # and Artifact Registry fails the final index upload with a bare "Internal
  # Server Error" after every layer has gone up. Without them the build is one
  # plain image manifest — the same digest, minus the attestation.
  # ${arr[@]+...} keeps bash 3.2 from treating an empty array as unbound.
  docker build --platform linux/amd64 --provenance=false --sbom=false \
    ${build_args[@]+"${build_args[@]}"} \
    --file "$train_dir/docker/Dockerfile" --tag "$local_image" "$train_dir"
else
  docker image inspect "$local_image" >/dev/null 2>&1 ||
    die "no local image $local_image to push; run without --skip-build"
fi

echo "==> Configuring Docker authentication for Artifact Registry (profile $profile)"
databricks auth docker configure --profile "$profile"
registry_host="$(databricks auth docker host --profile "$profile" | sed -n 's/^Registry host: //p')"
[[ -n "$registry_host" ]] ||
  die "could not read the registry host from 'databricks auth docker host --profile $profile'"

remote_image="$registry_host/$artifact_name:$tag"
echo "==> Pushing $remote_image"
docker tag "$local_image" "$remote_image"
# An image built elsewhere (or by an older version of this script) can still be
# an OCI index; push just its amd64 manifest so the registry never sees the
# index. The Descriptor field only exists on Docker's containerd image store,
# the only store that produces indexes.
push_args=()
if [[ "$(docker image inspect --format '{{if .Descriptor}}{{.Descriptor.MediaType}}{{end}}' "$local_image")" == *image.index* ]]; then
  push_args=(--platform linux/amd64)
fi
docker push ${push_args[@]+"${push_args[@]}"} "$remote_image"

echo "==> Versions of $artifact_name in Artifact Registry"
databricks api get --profile "$profile" \
  "/api/2.1/unity-catalog/software-artifacts/$artifact_name/versions" ||
  echo "warning: could not list versions (that needs READ METADATA); the push itself succeeded" >&2

echo
echo "Pushed $image_ref — train/train.yaml runs it as environment.unity_catalog_image."
echo "Next: databricks air run --file train/train.yaml --dry-run --profile $profile"
