#!/usr/bin/env bash
  
set -e

python -c 'from pyscf.dft import libxc; print(libxc)'
python -c 'from pyscf.dft.numint import libxc; print(libxc)'

cd ./pyscf
pytest -k 'not _slow'
