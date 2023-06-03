# makemore part 2 and 3 from Pyton to rust

This repo contains a rewrite in rust of the first two parts of the series (bgram and mlp) of [karpathy](https://github.com/karpathy/makemore) about makemore. The goal is to illustrate how to rewrite basic nn from py to rs using a thin wrapper around pytorchC++ (tch crate)

## Instructions

Only tested under linux with NVIDIA driver with manual install

### Manual install

* First install `just`

```sh
cargo install just
```

* Then populate your env

```bash
export LIBTORCH=$PWD/libtorch
export LIBTORCH_LIB=$LIBTORCH
export LIBTORCH_INCLUDE=$LIBTORCH
export TORCH_CUDA_VERSION=cu118
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

For fish user

```sh
set -lx LIBTORCH $PWD/libtorch 
set -lx LIBTORCH_LIB $LIBTORCH 
set -lx LIBTORCH_INCLUDE $LIBTORCH
set -lx TORCH_CUDA_VERSION cu118 
set -lx LD_LIBRARY_PATH $LIBTORCH/lib
```

* Then prepare build and run

```sh
just prepare
just build
just bgram
just mlp
```

### Automatic install

If you doesn't have nvidia-cuda gpu compatible driver, then to modify the `Cargo.toml` and uncomment `toch-sys` line. The `torch-sys` build will scan your system and do the job accordinly.

```toml
...
tch = "0.13.0"
torch-sys = { version = "0.13", features = ["download-libtorch"] }
```

Then build and run

```sh
just build
just bgram
just mlp
```

