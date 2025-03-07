#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
#
import pyscf
import numpy as np

import time
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_
from pyscf.mcdcft.dcfnal import dcfnal

def kernel(mc, dcxc, root=-1, **kwargs):
    ''' Calculate MC-DCFT total energy from a converged MCSCF wave function

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not run the CASSCF or CASCI calculation itself
                prior to calculating the MC-DCFT energy. Call mc.kernel() before passing to this function!
            dcxc : an instance of dcfnal class

        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-DCFT results (i.e., using state-averaged density matrices)

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())

    mc_1root = mc
    if isinstance(mc, StateAverageMCSCFSolver) and root >= 0:
        mc_1root = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
        mc_1root.fcisolver = fci.solver(mc._scf.mol, singlet=False, symm=False)
        mc_1root.mo_coeff = mc.mo_coeff
        mc_1root.ci = mc.ci[root]
        mc_1root.e_tot = mc.e_states[root]
    natorb, _, occ = mc_1root.cas_natorb_(sort=False)
    dm1s = np.asarray(mc_1root.make_rdm1s())
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer(dcxc, 'rdms', *t0)

    e_nuc = mc._scf.energy_nuc()
    h = mc._scf.get_hcore()
    dm1 = dm1s[0] + dm1s[1]
    vj = mc._scf.get_j(dm=dm1s)
    vj = vj[0] + vj[1]
    e_1e = np.dot(h.ravel(), dm1.ravel())
    e_coul = np.dot(vj.ravel(), dm1.ravel()) * 0.5

    e_mcscf = mc.e_mcscf
    try:
        e_mcscf = e_mcscf[root]
    except IndexError:
        pass
    e_ncwfn = e_mcscf - e_nuc - e_1e - e_coul
    t0 = logger.timer(dcxc, 'e_nuc, e_1e, e_coul, e_ncwfn', *t0)
    e_tot, E_dc = dcft_energy(dcxc, e_nuc, e_1e, e_coul, e_ncwfn, natorb, occ, **kwargs)

    chkdata = {'e_nuc':e_nuc, 'e_1e':e_1e, 'e_coul':e_coul, 'e_ncwfn':e_ncwfn,
               'spin':spin, 'natorb':natorb, 'occ':occ}

    return e_tot, E_dc, chkdata

def dcft_energy(dcxc, e_nuc, e_1e, e_coul, e_ncwfn, natorb, occ, **kwargs):
    ''' Calculate MC-DCFT total energy from decomposed energy components of one state

        Args:
            dcxc : an instance of dcfnal class
            e_nuc : float
                nuclear repulsion energy
            e_1e : float
                sum of kinetic energy and nuclear-electron attraction energy
            e_coul : float
                classical Coulomb energy
            e_ncwfn : float
                non-classical energy of the MCSCF wave function
            natorb : ndarray of shape (nao, nao)
                natural orbital of the MCSCF wave function
            occ : ndarray of shape (nao,)
                natural orbital occupation number of the MCSCF wave function

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    logger.debug(dcxc, 'CAS energy decomposition:')
    logger.debug(dcxc, 'e_nuc = %s', e_nuc)
    logger.debug(dcxc, 'E_1e = %s', e_1e)
    logger.debug(dcxc, 'e_coul = %s', e_coul)
    logger.debug(dcxc, 'e_ncwfn = %s', e_ncwfn)

    E_dc = get_E_dc(dcxc, natorb, occ, **kwargs)
    t0 = logger.timer(dcxc, 'E_dc', *t0)
    e_tot = e_nuc + e_1e + e_coul + dcxc.hyb_x * e_ncwfn + (1. - dcxc.hyb_x) * E_dc
    logger.note(dcxc, 'MC-DCFT E = %s, Edc(%s) = %s', e_tot, dcxc.display_name, E_dc)
    return e_tot, E_dc

def _recalculate_with_xc(dcxc, chkdata, **kwargs):
    ''' Recalculate MC-DCFT total energy based on intermediate quantities from a previous MC-DCFT calculation

        Args:
            dcxc : an instance of density coherence functional class
            chkdata : chkdata dict generated by previous calculation

        Returns:
            Total MC-DCFT energy including nuclear repulsion energy.
    '''
    e_nuc = chkdata['e_nuc']
    e_1e = chkdata['e_1e']
    e_coul = chkdata['e_coul']
    e_ncwfn = chkdata['e_ncwfn']
    natorb = chkdata['natorb']
    occ = chkdata['occ']

    return dcft_energy(dcxc, e_nuc, e_1e, e_coul, e_ncwfn, natorb, occ, **kwargs)


def get_E_dc(dcxc, natorb, occ, max_memory=20000, hermi=1):
    ''' E_MCDCFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_dc[1rdm]
        or, in other terms,
        E_MCDCFT = T_MCSCF[rho] + E_ext[rho] + E_coul[rho] + E_dc[1rdm]
                 = E_DFT[1rdm] - E_xc[rho] + E_dc[1rdm]
        Args:
            dcxc : an instance of dcfnal class
            natorb : ndarray of shape (nao, nao)
                natural orbital of the MCSCF wave function
            occ : ndarray of shape (nao,)
                natural orbital occupation number of the MCSCF wave function

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-DCFT density-coherence exchange-correlation energy

    '''
    ni, dens_deriv = dcxc._numint, dcxc.dens_deriv
    norbs_ao = natorb.shape[0]

    E_dc = 0.0
    dcxc.ms = 0.0

    t0 = (logger.process_clock(), logger.perf_counter())
    for ao, mask, weight, coords in ni.block_loop(dcxc.mol, dcxc.grids, norbs_ao, dens_deriv, max_memory):
        E_dc += dcxc.get_E_dc(natorb, occ, ao, weight)
        t0 = logger.timer(dcxc, 'on-top exchange-correlation energy calculation', *t0)

    return E_dc

class DCFT_base(lib.StreamObject):
    def __init__(self, mol, dc=None, grids_level=None, verbose=None):
        if verbose is None:
            verbose = mol.verbose
        self.verbose = verbose
        if dc is None:
            self.dcfnal = None
            self.mol = mol
        else:
            self.dcfnal = dcfnal(mol, dc, grids_level, verbose=verbose)
        self.grids_level = grids_level

    def load_mcdcft_chk(self, chkfile):
        self.chkdata = lib.chkfile.load(chkfile, 'mcdcft')

    def recalculate_with_xc(self, dc, chkdata=None,
                            load_chk=None, dump_chk=None, grids_level=None, **kwargs):
        ''' Recalculate MC-DCFT total energy based on intermediate quantities from a previous MC-DCFT calculation

            Args:
                dc : str or dcfnal instance representing the density coherence functional.
                     The str should follow the convention in the ks module.
                     i.e. do NOT add any prefix
                chkdata : chkdata dict generated by previous calculation
                load_chk : str of chk filename to load chkdata from before the calculation
                dump_chk : str of chk filename to dump newly calculated energies
                grids_level : int to be passed to grids.levels
                     or a tuple of int to be passed to grids.atom_grid

            Returns:
                Total MC-DCFT energy including nuclear repulsion energy and E_dc
        '''
        if grids_level is None:
            grids_level = self.grids_level
        if load_chk is not None:
            self.load_mcdcft_chk(load_chk)
        if chkdata is None:
            chkdata = self.chkdata
        if isinstance(dc, str):
            self.dcfnal = dcfnal(self.mol, dc, grids_level, self.verbose)
        else:
            self.dcfnal = dc
        n_states = chkdata['n_states']
        if n_states > 1:
            epdft = [_recalculate_with_xc(self.dcfnal, ichkdata) for ichkdata in chkdata]
            self.e_states, self.e_dc = zip(*epdft)
            weights = chkdata['weights']
            self.e_tot = np.dot(self.e_states, weights)
        else:
            self.e_tot, self.e_dc = _recalculate_with_xc(self.dcfnal, chkdata)
        if dump_chk is not None:
            lib.chkfile.dump(dump_chk, 'mcdcft/e_tot/' + self.dcfnal.display_name, self.e_tot)
            lib.chkfile.dump(dump_chk, 'mcdcft/e_dc/' + self.dcfnal.display_name, self.e_dc)
        return self.e_tot, self.e_dc

    def kernel(self, mo_coeff=None, ci=None, skip_scf=False, **kwargs):
        if not skip_scf:
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = super().kernel(mo_coeff,
                                                                                              ci, **kwargs)

        # TODO: State average has not been tested !!!
        if isinstance(self, StateAverageMCSCFSolver):
            epdft = [kernel(self, self.dcfnal, root=ix, **kwargs) for ix in range(len(self.e_states))]
            self.e_mcscf = self.e_states
            #  self.fcisolver.e_states = [e_tot for e_tot, e_dc in epdft]
            #  self.e_dc = [e_dc for e_tot, e_dc in epdft]
            self.fcisolver.e_states, self.e_dc, self.chkdata = zip(*epdft)
            self.chkdata['n_states'] = len(epdft)
            self.chkdata['weights'] = self.weights
            self.e_tot = np.dot(self.e_states, self.weights)
        else:
            self.e_tot, self.e_dc, self.chkdata = kernel(self, self.dcfnal, **kwargs)
            self.chkdata['n_states'] = 1
        self.chkdata['e_tot'] = {self.dcfnal.display_name: self.e_tot}
        if self.chkfile is not None:
            self.dump_mcdcft_chk(self.chkfile)
        return self.e_tot, self.e_dc, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def dump_mcdcft_chk(self, chkfile, key='mcdcft', chkdata=None):
        '''Save MC-DCFT calculation results in chkfile.
        '''
        if chkdata is None:
            chkdata = self.chkdata
        lib.chkfile.dump(chkfile, key, chkdata)
        lib.chkfile.dump_mol(self.mol, chkfile)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info('density coherence exchange-correlation functional: %s', self.dcfnal.dc_code)

    # TODO: gradient has not been implemented
    def nuc_grad_method(self):
        raise NotImplementedError("MC-DCFT nuclear gradients")

    @property
    def dc_code(self):
        return self.dcfnal.dc_code


def get_mcdcft_child_class(mc, dc, ncas, nelecas, **kwargs):
    class CASDCFT(DCFT_base, mc.__class__):
        def __init__(self, mol, dc, ncas, nelecas, **kwargs):
            try:
                mc.__class__.__init__(self, mol, ncas, nelecas)
            except TypeError:
                mc.__class__.__init__(self)
            DCFT_base.__init__(self, mol, dc, **kwargs)
            keys = set(('e_dc', 'e_mcscf', 'e_states'))
            self._keys = set(self.__dict__.keys()).union(keys)

    dcft = CASDCFT(mc.mol, dc, ncas, nelecas, **kwargs)
    dcft.__dict__.update(mc.__dict__)
    return dcft

def CASSCFDCFT(mf_or_mol, dc, ncas, nelecas, grids_level=None,
               chkfile=None, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASSCF(mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    if chkfile is not None:
        mc.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
    mc = get_mcdcft_child_class(mc, dc, ncas, nelecas,
                                grids_level=grids_level, **kwargs)
    mc.chkfile = chkfile
    return mc

CASSCF = CASSCFDCFT
