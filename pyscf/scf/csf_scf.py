#!/usr/bin/env python

from functools import reduce
import numpy as np
import pyscf.lib.logger as logger
from pyscf import lib
from pyscf.mcscf import casci, mc1step, mc_ao2mo
from pyscf.fci import direct_spin1, cistring
from pyscf.soscf import newton_ah


# gradients, hessian operator and hessian diagonal
def gen_g_hop(csfmf, mo, u, casdm1, casdm2, eris):
    ncas = csfmf.ncas
    ncore = csfmf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    dm1 = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    dm_core = mo[:,:ncore].dot(mo[:,:ncore].conj().T) * 2

    jkcaa = np.empty((nocc,ncas))
    vhf_a = np.empty((nmo,nmo))
    # ~ (J + 2K)
    dm2tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3)
    dm2tmp = dm2tmp.reshape(ncas**2,-1)
    hdm2 = np.empty((nmo,ncas,nmo,ncas))
    g_dm2 = np.empty((nmo,ncas))
    vj_c, vk_c = csfmf.get_jk(mol, dm_core)
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        if i < nocc:
            jkcaa[i] = np.einsum('ik,ik->i', 6*kbuf[:,i]-2*jbuf[i], casdm1)
        vhf_a[i] =(np.einsum('quv,uv->q', jbuf, casdm1) -
                   np.einsum('uqv,uv->q', kbuf, casdm1) * .5)
        jtmp = lib.dot(jbuf.reshape(nmo,-1), casdm2.reshape(ncas*ncas,-1))
        jtmp = jtmp.reshape(nmo,ncas,ncas)
        ktmp = lib.dot(kbuf.transpose(1,0,2).reshape(nmo,-1), dm2tmp)
        hdm2[i] = (ktmp.reshape(nmo,ncas,ncas)+jtmp).transpose(1,0,2)
        g_dm2[i] = np.einsum('uuv->v', jtmp[ncore:nocc])
    jbuf = kbuf = jtmp = ktmp = dm2tmp = None
    vhf_c = mo.conj().T.dot(vj_c - vk_c*.5).dot(mo)
    vhf_ca = vhf_c + vhf_a
    h1e = csfmf.get_hcore()
    h1e_mo = mo.T.dot(h1e).dot(mo)

    ################# gradient #################
    g = np.zeros_like(h1e_mo)
    g[:,:ncore] = (h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
    g[:,ncore:nocc] = np.dot(h1e_mo[:,ncore:nocc]+vhf_c[:,ncore:nocc],casdm1)
    g[:,ncore:nocc] += g_dm2

    ############## hessian, diagonal ###########
    h_diag = np.einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    idx = np.arange(nmo)
    h_diag[idx,idx] += g_diag * 2

    v_diag = vhf_ca.diagonal() # (pr|kl) * E(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1) * 2
    h_diag[:ncore] += v_diag * 2
    idx = np.arange(ncore)
    h_diag[idx,idx] -= v_diag[:ncore] * 4
    # V_{pr} E_{sq}
    tmp = np.einsum('ii,jj->ij', vhf_c, casdm1)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -vhf_c[ncore:nocc,ncore:nocc] * casdm1
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # -2(pr|sq) + 4(pq|sr) + 4(pq|rs) - 2(ps|rq)
    tmp = 6 * eris.k_pc - 2 * eris.j_pc
    h_diag[ncore:,:ncore] += tmp[ncore:]
    h_diag[:ncore,ncore:] += tmp[ncore:].T

    # -(qr|kp) E_s^k  p in core, sk in active
    h_diag[:nocc,ncore:nocc] -= jkcaa
    h_diag[ncore:nocc,:nocc] -= jkcaa.T

    v_diag = np.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag.T
    h_diag[:,ncore:nocc] += v_diag

    g_orb = csfmf.pack_uniq_var(g-g.T)
    h_diag = csfmf.pack_uniq_var(h_diag)

    def h_op(x):
        x1 = csfmf.unpack_uniq_var(x)

        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(lib.dot, (h1e_mo, x1, dm1))
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        x2 -= np.dot((g+g.T), x1) * .5
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += reduce(np.dot, (x1[:ncore,ncore:], vhf_ca[ncore:])) * 2
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(np.dot, (casdm1, x1[ncore:nocc], vhf_c))
        x2[:,ncore:nocc] += np.einsum('purv,rv->pu', hdm2, x1[:,ncore:nocc])

        if ncore > 0:
            # Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            #    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            # x2[:,:ncore] += H * x1[:,:ncore] => (becuase x1=-x1.T) =>
            # x2[:,:ncore] += -H' * x1[:ncore] => (becuase x2-x2.T) =>
            # x2[:ncore] += H' * x1[:ncore]
            va, vc = csfmf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc

        # (pr<->qs)
        x2 = x2 - x2.T
        return csfmf.pack_uniq_var(x2)

    return g_orb, None, h_op, h_diag


def kernel(csfmf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    log = logger.new_logger(csfmf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    mo = mo_coeff
    nmo = mo.shape[1]
    ncore = csfmf.ncore
    ncas = csfmf.ncas
    nocc = ncore + ncas
    nelecas = csfmf.nelecas
    eris = csfmf.ao2mo(mo)
    e_tot = csfmf.energy_tot(mo)

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(csfmf, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    elast = e_tot
    r0 = None
    casdm1, casdm2 = direct_spin1.make_rdm12(csfmf.csf, ncas, nelecas)

    cycle = 0
    norm_ddm = 0
    for cycle in range(csfmf.max_cycle):
        max_stepsize = csfmf.max_stepsize
        rota = csfmf.rotate_orb_cc(mo, lambda:None, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad*.3, max_stepsize, log)
        u, g_orb, njk1, r0 = next(rota)
        rota.close()
        norm_t = np.linalg.norm(u-np.eye(nmo))
        norm_gorb = np.linalg.norm(g_orb)

        mo = csfmf.rotate_mo(mo, u, log)
        e_tot = csfmf.energy_tot(mo)
        de, elast = e_tot - elast, e_tot
        log.info('cycle= %d  E= %.15g  dE=%5.3g  |u-1|=%5.3g  |g[o]|=%5.3g',
                 cycle, e_tot, de, norm_t, norm_gorb)

        if abs(de) < tol and norm_gorb < conv_tol_grad:
            conv = True
            break
        else:
            elast = e_tot

        eris = None
        eris = csfmf.ao2mo(mo)

    if conv:
        log.info('CSF-SCF converged')
    else:
        log.info('CSF-SCF not converged')
    return conv, e_tot, mo

def energy_tot(csfmf, mo_coeff):
    ncas = csfmf.ncas
    nelecas = csfmf.nelecas
    eri_cas = csfmf.get_h2eff(mo_coeff)
    h1eff, energy_core = csfmf.get_h1eff(mo_coeff)
    csf = csfmf.csf
    e_cas = direct_spin1.energy(h1eff, eri_cas, csf, ncas, nelecas)
    e_tot = e_cas + energy_core
    logger.debug(csfmf, 'Etot = %s', e_tot)
    return e_tot


class CSFSCF:

    max_stepsize         = mc1step.CASSCF.max_stepsize
    max_cycle            = mc1step.CASSCF.max_cycle_macro
    max_cycle_micro      = mc1step.CASSCF.max_cycle_micro
    conv_tol             = mc1step.CASSCF.conv_tol
    conv_tol_grad        = mc1step.CASSCF.conv_tol_grad
    ah_level_shift       = mc1step.CASSCF.ah_level_shift
    ah_conv_tol          = mc1step.CASSCF.ah_conv_tol
    ah_max_cycle         = mc1step.CASSCF.ah_max_cycle
    ah_lindep            = mc1step.CASSCF.ah_lindep
    ah_start_tol         = mc1step.CASSCF.ah_start_tol
    ah_start_cycle       = mc1step.CASSCF.ah_start_cycle
    ah_grad_trust_region = mc1step.CASSCF.ah_grad_trust_region
    kf_interval          = mc1step.CASSCF.kf_interval
    kf_trust_region      = mc1step.CASSCF.kf_trust_region

    def __init__(self, mf, csf):
        mol = mf.mol
        self._scf = mf
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.mol = mol
        ncas, nelecas, csf = _csf2fcivec(csf)
        self.csf = csf
        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = mol.nelec[0] - nelecas[0]
        self.frozen = 0

        self.e_tot = None
        self.e_cas = None
        self.mo_coeff = mf.mo_coeff
        self.converged = False

    def ao2mo(self, mo):
        mol = self.mol
        nao, nmo = mo.shape
        ncore = self.ncore
        ncas = self.ncas

        mem_now = lib.current_memory()[0]
        max_memory = max(3000, self.max_memory*.9-mem_now)
        log = logger.new_logger(self, self.verbose)
        self.feri = lib.H5TmpFile()
        eris = lambda: None
        eris.j_pc, eris.k_pc = mc_ao2mo.trans_e1_outcore(
            mol, mo, ncore, ncas, self.feri,
            max_memory=max_memory, level=0, verbose=log)
        eris.ppaa = self.feri['ppaa']
        eris.papa = self.feri['papa']
        return eris

    def uniq_var_indices(self, nmo, ncore, ncas, *args, **kwargs):
        nocc = ncore + ncas
        mask = np.zeros((nmo,nmo),dtype=bool)
        mask[ncore:,:nocc] = True
        mask[ncore:nocc,ncore:nocc][np.triu_indices(ncas)] = False
        return mask

    def kernel(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff

        self.converged, self.e_tot, self.mo_coeff = kernel(
            self, mo_coeff, tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
            verbose=self.verbose)
        logger.note(self, 'CSF-SCF energy = %#.15g', self.e_tot)
        return self.e_tot

    energy_tot = energy_tot
    gen_g_hop = gen_g_hop
    rotate_orb_cc = mc1step.rotate_orb_cc

    pack_uniq_var = mc1step.CASSCF.pack_uniq_var
    unpack_uniq_var = mc1step.CASSCF.unpack_uniq_var
    update_rotate_matrix = mc1step.CASSCF.update_rotate_matrix
    update_jk_in_ah = mc1step.CASSCF.update_jk_in_ah
    rotate_mo = mc1step.CASSCF.rotate_mo

    energy_nuc = casci.CASCI.energy_nuc
    get_hcore = casci.CASCI.get_hcore
    get_h1eff = casci.CASCI.h1e_for_cas
    get_h2eff = casci.CASCI.ao2mo
    get_veff = casci.CASCI.get_veff
    get_jk = casci.CASCI.get_jk

def _csf2fcivec(csf_dic):
    dets_conf = csf_dic.keys()
    dets_a = np.array([det[0] for det in dets_conf], dtype=int)
    dets_b = np.array([det[1] for det in dets_conf], dtype=int)
    assert dets_a.ndim == dets_b.ndim == 2
    neleca = dets_a.shape[1]
    nelecb = dets_b.shape[1]
    ndet = dets_a.shape[0]

    nocc = max(dets_a.max(initial=0), dets_b.max(initial=0)) + 1
    ncore = min(dets_a.max(initial=nocc), dets_b.min(initial=nocc))
    ncas = nocc - ncore
    assert ncas < 16

    strings_a = np.zeros(ndet, dtype=np.int64)
    strings_b = np.zeros(ndet, dtype=np.int64)
    for str_bits in 1 << (dets_a.T - ncore):
        strings_a = np.bitwise_or(strings_a, str_bits)
    for str_bits in 1 << (dets_b.T - ncore):
        strings_b = np.bitwise_or(strings_b, str_bits)
    addr_a = cistring.strs2addr(ncas, neleca, strings_a)
    addr_b = cistring.strs2addr(ncas, nelecb, strings_b)

    na = cistring.num_strings(ncas, neleca)
    nb = cistring.num_strings(ncas, nelecb)
    civec = np.zeros((na, nb))
    civec[(addr_a, addr_b)] = list(csf_dic.values())
    return ncas, (neleca, nelecb), civec


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; H 0 0 1.3'
    mol.basis = '631g'
    mol.verbose = 4
    mol.build()

    mf = mol.RHF().run()

    # 2-determinant CSF, each determinant is labelled by two lists of orbitals
    # for alpha electrons and beta electrons separately
    csf = {
        ((2,), (3,)): .5**.5,
        ((3,), (2,)): -.5**.5
    }
    csfmf = CSFSCF(mf, csf)
    e_tot = csfmf.kernel()
