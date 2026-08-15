import numpy as np

#A single qubit gate is a 2x2 unitary matrix. This works on an input state of a single qubit to produce an output state.

#U_gate is the most general single qubit gate. All other single qubit gates can be written in this form.
def U_gate(theta,phi,lamda):

    U=np.zeros((2,2))+0*1j
    # 1j is the python way of writing i, the complex number. Adding 0*1j = 0*i, adds a 0 complex part, making the datatype of the
    # matrix complex. This is often safer to do, to later avoid issues about neglecting imaginary parts, which might happen otherwise.

    U[0,0]=np.cos(theta/2)
    U[0,1]=-np.exp(1j*lamda)*np.sin(theta/2)
    U[1,0]=np.exp(1j*phi)*np.sin(theta/2)
    U[1,1]=np.exp(1j*phi+1j*lamda)*np.cos(theta/2)

    return U



# Hadamard is equivalent to U_gate(np.pi/2,0,np.pi)
def Hadamard():

    H=np.zeros((2,2))+0*1j

    H[0,0]=1
    H[0,1]=1
    H[1,0]=1
    H[1,1]=-1

    H=H/2**0.5

    return H


# X_gate is sigmax Pauli matrix, equivalent to U_gate(np.pi,0,np.pi)
def X_gate():

    X=np.array([[0,1],[1,0]]) + 0*1j

    return X

# Y_gate is sigmay Pauli matrix, equivalent to U_gate(np.pi,np.pi/2,np.pi/2)
def Y_gate():

    Y=np.array([[0,-1j],[1j,0]]) + 0*1j

    return Y

# Z_gate is sigmaz Pauli matrix, equivalent to U_gate(0,np.pi,0)
def Z_gate():

    Z=np.array([[1,0],[0,-1]]) + 0*1j

    return Z

# P_gate is equivalent to U_gate(0,0,lamda)
def P_gate(lamda):

    P=np.array([[1,0],[0,np.exp(1j*lamda)]])

    return P

#S_gate is P_gate(np.pi/2)
#T_gate is P_gate(np.pi/4)
#I_gate is U(0,0,0) or P(0)


    
def SQRT_X_gate():

    m=X_gate()

    eigvals,eigvecs=np.linalg.eigh(m)

    a=np.diag([np.exp(-1j*np.pi/2),1])

    gate= eigvecs @ a @ eigvecs.T.conj()

    return gate


def SQRT_Y_gate():

    m=Y_gate()

    eigvals,eigvecs=np.linalg.eigh(m)

    a=np.diag([np.exp(-1j*np.pi/2),1])

    gate= eigvecs @ a @ eigvecs.T.conj()

    return gate


def T_gate():

    gate=np.zeros((2,2))+0*1j

    gate[0,0]=1
    gate[1,1]=np.exp(-1j*np.pi/4)

    return gate


# ---------------------------------------------------------------------------
# Rotation gates  R_P(theta) = exp(-i theta P / 2)   (generators are P/2).
# ---------------------------------------------------------------------------

def Rx(theta):
    """Rx(theta) = exp(-i theta X/2).

    [[cos(theta/2), -i sin(theta/2)], [-i sin(theta/2), cos(theta/2)]].
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta):
    """Ry(theta) = exp(-i theta Y/2).

    [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta):
    """Rz(theta) = exp(-i theta Z/2).

    diag(exp(-i theta/2), exp(+i theta/2)). Note: differs from P_gate(theta)
    by the global phase exp(+i theta/2); use Rz for generator consistency.
    """
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def U3(theta, phi, lam):
    """General single-qubit gate U3(theta, phi, lam).

    Same parametrisation as U_gate above, re-exported under the standard name
    with a guaranteed-complex dtype.
    """
    return np.array(
        [[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
         [np.exp(1j * phi) * np.sin(theta / 2),
          np.exp(1j * (phi + lam)) * np.cos(theta / 2)]],
        dtype=complex,
    )
