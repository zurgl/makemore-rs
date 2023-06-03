# just manual: https://github.com/casey/just#readme

_default:
  just --list

build:
  cd rust && cargo build

clean:
  cd rust && cargo clean

prepare:
  curl -o libtorch.zip -L \
    https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
  unzip libtorch.zip
  rm libtorch.zip

bgram:
  cd rust && cargo run --bin bgram

mlp:
  cd rust && cargo run --bin mlp -- gpu