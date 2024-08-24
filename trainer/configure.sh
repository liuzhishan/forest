if [ -f .bazelrc ]; then
  rm -f .bazelrc
fi

PYTHON=python
if [[ "$#" -gt 0 ]]; then
  PYTHON=$1
fi

if $PYTHON -c "import tensorflow as tf" &> /dev/null; then
    echo 'using installed tensorflow'
else
    echo "ERROR: You must instal tensorflow first !!!"
fi
$PYTHON tools/build/configure.py
