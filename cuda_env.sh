#!/bin/sh
# Source this script to set up CUDA environment for running with GPU feature.
#   source ./cuda_env.sh
#   cargo run --features gpu -- --prompt "hello"
export CUDA_PATH="$(dirname "$0")/cuda_local"
export LD_LIBRARY_PATH="$(dirname "$0")/cuda_local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "CUDA_PATH=$CUDA_PATH"
echo "LD_LIBRARY_PATH includes $CUDA_PATH/lib"
