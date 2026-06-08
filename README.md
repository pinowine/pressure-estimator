# pressure-estimator

A lightweight pressure estimation toolkit for Tetris-like games.

The project is still early, but the core idea is simple: collect a compact set of board and player-behavior signals, pass them through an exported ONNX model, and return a normalized pressure score that can be used by tools, bots, or game UI.

## Runtime device

The estimator supports three runtime modes:

- `auto`: prefer CUDA when ONNX Runtime exposes it, then fall back to CPU.
- `cpu`: force `CPUExecutionProvider`, useful for laptops, servers without NVIDIA hardware, or predictable local debugging.
- `cuda`: force `CUDAExecutionProvider`, useful when the target machine has a compatible NVIDIA CUDA stack and needs better inference throughput.
