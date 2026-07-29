import numpy as np


#Controlled X gate (also called C-NOT or Controlled NOT gate
# )
def CX_gate():

    CX=np.zeros((4,4))+0*1j

    CX[0,0]=1
    CX[1,1]=1
    CX[2,3]=1
    CX[3,2]=1

    return CX

#Controlled Y gate
def CY_gate():

    CY=np.zeros((4,4))+0*1j

    CY[0,0]=1
    CY[1,1]=1
    CY[2,3]=-1j
    CY[3,2]=1j

    return CY

#Controlled Z gate
def CZ_gate():

    CZ=np.zeros((4,4))+0*1j

    CZ[0,0]=1
    CZ[1,1]=1
    CZ[2,2]=1
    CZ[3,3]=-1

    return CZ

def SWAP_gate():

    swap=np.zeros((4,4))+0*1j

    swap[0,0]=1
    swap[3,3]=1
    swap[1,2]=1
    swap[2,1]=1

    return swap

def PSWAP_gate(theta):

    S=SWAP_gate()

    I4=np.eye(4)

    PS=np.cos(theta)*I4 - 1j*np.sin(theta)*S

    return PS


# ---------------------------------------------------------------------------
# Two-qubit rotations / phase gates for the RC-label study (CLAUDE.md §2.5, §4).
# Basis order is big-endian |q1 q2>: {|00>, |01>, |10>, |11>} = indices 0..3,
# matching operator_on_state.two_qubit_op.
# ---------------------------------------------------------------------------

def Rzz(theta):
    """Rzz(theta) = exp(-i theta (Z x Z) / 2)  (CLAUDE.md §2.5).

    Z x Z = diag(1, -1, -1, 1), so
    Rzz = diag(e^{-i theta/2}, e^{+i theta/2}, e^{+i theta/2}, e^{-i theta/2}).
    The 'hard' gate that carries the coherent error in the TFIM Trotter step
    (hard gate = Rzz(2 tau)).
    """
    a = np.exp(-1j * theta / 2)
    b = np.exp(1j * theta / 2)
    return np.diag([a, b, b, a]).astype(complex)


def CZ(theta=None):
    """Controlled-Z = diag(1, 1, 1, -1) (CLAUDE.md §4).

    Alias of CZ_gate() under the spec name. `theta` is accepted and ignored so
    callers can use CZ() uniformly; use CPhase(theta) for a tunable phase.
    """
    return CZ_gate()


def CPhase(theta):
    """Controlled-phase gate CPhase(theta) = diag(1, 1, 1, e^{i theta}) (CLAUDE.md §4).

    CZ is the special case theta = pi.
    """
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


