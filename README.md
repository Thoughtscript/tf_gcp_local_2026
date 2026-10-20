# tf_gcp_local

 [![agents-cli 1.11.0](https://img.shields.io/badge/agents--cli-1.11.0-blue.svg)](https://google.github.io/agents-cli/guide/getting-started/) [![TensorFlow 2.20.0](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00)](https://github.com/tensorflow/tensorflow/releases/tag/v2.20.0)

Exploring [agents-cli](https://google.github.io/agents-cli/guide/getting-started/) and alternatives ([vyomi](https://vyomi.cloud/), [localgcp](https://localgcp.com/)):
1. To test and spin up some GCP compliant Gemini Enterprise Agent Platform environments without spending a lot of time, money.
1. To test various TensorFlow configurations and deployments (`logistic`, `linear`, `forecasting`, batch sizes, distributed gpu, sharded `TFRecords`, etc.) that come up in the GCP PMLE.
1. Opted for the official Google `agents-cli` since multi-Agent examples can be entirely Containerized, used locally, migrated to an actual GCP environment, and since it's likely used in various production enterprise GCP settings.
1. To review test patterns for GCP EAP Agent Models.

This is an ai-assisted-development effort using GitHub Copilot to quickly scaffold an `agents-cli` lab with a few kinds of tf examples (`logistic`, `linear`, `forecasting`).

## agents-cli Local Development

The inbuilt `agents-cli playground` [supports](https://google.github.io/agents-cli/cli/#playground) both local development/testing and publishing to GCP Gemini.

There are some limitations: I've chosen a configuration that's CPU only on a Mac. 
* For ARM GPU support on newer Macs, one has to use `tensorflow-metal` but it only provides a single GPU backend. It doesn't offer much advantage for this limited demo.
* I will probably tinker around with some kind of NVIDIA GPU with multiple Linux Containers in the future.

## TF Settings

Example batch sizes, distributed gpu, multiworker, sharded `TFRecords`, `memory.pin` configs can be found in [train.yaml](./tf-lab/app/ml/train.yaml).

## Running the Lab

Train Models:
```bash
docker compose run train --demo forecast --set data.num_shards=16 
docker compose run train --demo logistic --set data.num_shards=16 
docker compose run train --demo linear --set data.num_shards=16 

# Multiple works with global batch
docker compose --profile multiworker up --abort-on-container-exit
```

Spin up the Agent:
1. Get some [Google AI Studio credentials](https://aistudio.google.com/api-keys) and configure [.env](./tf-lab/.env).
    * This is probably the fastest, easiest, cheapest way to do this.
1. `docker compose --profile agent up` 
1. http://localhost:8080/dev-ui/
1. The Agent integrates with GCP Gemini 3.8 in the cloud and handles LLM interactions against tools defined [here](./tf-lab/app/agent.py).
1. Screenshots are supplied in [_lab](/_lab/).

# Resources and Links
1. https://google.github.io/agents-cli/guide/getting-started
1. https://vyomi.cloud
1. https://localgcp.com
1. https://google.github.io/agents-cli/cli/#playground
1. https://aistudio.google.com/api-keys