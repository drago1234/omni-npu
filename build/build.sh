set -e
BUILD_ROOT="$(dirname $(dirname "$(realpath "$0")"))"
cd $BUILD_ROOT/
pip install -e .