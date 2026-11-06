# third_party

The integration point for the official starter toolkit [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data/tree/release/0.2.0/).
The parts for Track 2 are in its `robotic_grounding/`.

An ADR will decide the integration method (git submodule, a local checkout at a fixed commit, or copies as needed). Until then,
the local checkout is in `third_party/video_to_data/`, and `.gitignore` excludes it.

We do not change the official source code. If a change is necessary, it goes here as a patch file. The file name or the file header gives the upstream version and the reason for the change.
