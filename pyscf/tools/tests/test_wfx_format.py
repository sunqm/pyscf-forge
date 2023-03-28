#!/usr/bin/env python
# Copyright 2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import tempfile
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.tools import wfx_format

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = '''
    N  0.0000000000   0.0000000000   0.0000000000
    N  0.0000000000   0.0000000000   1.0977000000
               '''
    mol.basis = 'sto-3g'
    mol.symmetry = 1
    mol.verbose = 0
    mol.build()

    mf = mol.RHF().run()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_from_chkfile(self):
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfcidump:
            wfx_format.from_chkfile(tmpfcidump.name, mf.chkfile)

    def test_from_scf(self):
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfcidump:
            wfx_format.from_scf(mf, tmpfcidump.name)

    def test_from_mo(self):
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfcidump:
            wfx_format.from_mo(mf.mol, tmpfcidump.name, mf.mo_coeff)

    def test_from_kscf(self):
        cell = pyscf.M(
            atom = '''He 2 0 0''',
            a = numpy.eye(3) * 3,
            mesh = [10,10,10],
            basis = { 'He': [[0, (0.8, 1.0)],
                             [0, (1.2, 1.0)]]})
        mf = cell.KRHF()
        mf.kpts = cell.make_kpts([2,1,1])
        mf.kernel()
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfcidump:
            wfx_format.from_scf(mf, tmpfcidump.name)


if __name__ == "__main__":
    print("Full Tests for wfx_format")
    unittest.main()
