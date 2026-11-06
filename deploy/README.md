# deploy

The runtime environment of the GPU components: Isaac Lab container images, GPU host setup, credential injection.

- The image is the only unit that runs. We do not install an environment on the host to run things.
- The environment that the official toolkit states is Isaac Lab 2.3.1, NVIDIA driver 580.126.09 / CUDA 13.0.
  RTX A6000 and L40S are verified. Blackwell does not support HOI reconstruction. These lines come from the official README. We did not test them in this repository.
- Credentials such as the Hugging Face token come in only through environment variables or stdin.

No content yet. The versions and the host sources wait for an ADR.
