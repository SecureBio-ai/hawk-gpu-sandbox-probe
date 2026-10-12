# Hawk GPU sandbox probe

A one-sample [Inspect](https://inspect.aisi.org.uk/) eval that checks GPU services in Kubernetes sandboxes work on an [Inspect Hawk](https://hawk.metr.org/) deployment. It is an infrastructure test, not an eval. It makes no real model call and contains no eval content.

It tests one pattern: a GPU-backed scoring or tool model runs as a service in each sample's sandbox, and Hawk's runner reaches it through Kubernetes `exec`, not over the network.

## What it checks

The sandbox ([`values.yaml`](src/gpu_sandbox_probe/values.yaml), [inspect_k8s_sandbox](https://github.com/UKGovernmentBEIS/inspect_k8s_sandbox) `agent-env` chart values) has two services:

- **`default`:** a small CPU service, where an agent would work.
- **`gpu`:** a PyTorch CUDA image that asks for one `nvidia.com/gpu` and is `networkIsolated`.

The scorer runs in Hawk's runner and execs into them:

| Check | Pass means |
| --- | --- |
| `gpu_visible` | `nvidia-smi` lists a GPU. It reports the driver version. |
| `gpu_torch_cuda` | PyTorch sees CUDA and multiplies two matrices on the GPU. It reports the torch and CUDA versions, the device name and its compute capability. |
| `gpu_listener_local` | A listener started in the `gpu` service answers inside that service. This is the control for the next check. |
| `default_to_gpu_blocked` | The `default` service can't connect to that listener, so an agent couldn't reach a scoring model. |
| `gpu_egress_blocked` | The `gpu` service can't reach the internet, so a model's weights have to be in its image. |

**On Hawk v3.7.2:**

- Hawk sees the GPU limit and adds the GPU taint toleration itself.
- Karpenter launches a node from the built-in `g4dn` (NVIDIA T4) pool.
- `runtimeClassName: CLUSTER_DEFAULT` leaves the pod on the node's default runtime, which the GPU operator sets to the nvidia runtime.

## Running it

You need:

- the `hawk` CLI, logged in to the target deployment;
- GPU quota for the deployment's GPU pool.

Hawk installs the probe straight from this public repository, so no token is needed:

```bash
sha=$(git ls-remote https://github.com/SecureBio-ai/hawk-gpu-sandbox-probe.git HEAD | cut -f1)
curl -s https://raw.githubusercontent.com/SecureBio-ai/hawk-gpu-sandbox-probe/$sha/gpu-sandbox.eval-set.yaml |
  sed "s/<COMMIT_SHA>/$sha/" > /tmp/gpu-sandbox.eval-set.yaml

export HAWK_API_URL=<your deployment's API URL>
hawk eval-set /tmp/gpu-sandbox.eval-set.yaml     # prints the eval-set ID
hawk watch <eval-set-id>

# The scorer's table of checks:
hawk list samples <eval-set-id> --json | jq -r '.. | .explanation? // empty'
```

**Expect a few minutes before the sample starts:** Karpenter launches a GPU node and the node pulls a 3 GB image. A g4dn.xlarge costs about $0.53 an hour, so a run costs well under a dollar. Hawk deletes the sandbox when the sample ends.

**If it stalls,** run `hawk status <eval-set-id>`. A `gpu` pod that stays Pending usually means no GPU quota or no capacity in the cluster's zones. `hawk delete <eval-set-id>` tears everything down.

## License

Copyright © 2026 SecureBio. All rights reserved.
