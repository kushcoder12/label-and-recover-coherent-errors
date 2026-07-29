import numpy as np


def single_qubit_op(state, operator, p, n):
    final_s = np.zeros((2**n,), dtype=complex)

    for r in range(2**n):
        r_bar = 2 ** (n - p) ^ r

        q = int((2 ** (n - p) & r) / (2 ** (n - p)))

        q_bar = q ^ 1
        # print(r_bar, q, q_bar, r, p)
        final_s[r] = state[r] * operator[q, q] + state[r_bar] * operator[q, q_bar]

    return final_s


# import numpy as np


# def single_qubit_op(state, operator, p, n):
#     final_s = np.zeros((2**n,), dtype=complex)

#     # Bitmask for the p-th qubit (0-indexed from left)
#     bitmask = 1 << (n - p - 1)

#     for r in range(2**n):
#         # Flip p-th qubit
#         r_bar = r ^ bitmask

#         # Extract value of p-th qubit (0 or 1)
#         q = (r & bitmask) >> (n - p - 1)

#         # Opposite bit
#         q_bar = q ^ 1

#         final_s[r] = state[r] * operator[q, q] + state[r_bar] * operator[q, q_bar]

#     return final_s


def two_qubit_op(state, operator, s1, s2, n):
    final_s = np.zeros((2**n,), dtype=complex)

    for r in range(2**n):
        n1 = 2 ** (n - s1)
        n2 = 2 ** (n - s2)
        a1 = int((n1 & r) / n1)
        a2 = int((n2 & r) / n2)
        n3 = n1 + n2
        l = 2 * a1 + a2
        r3 = n3 | r
        r2 = n2 ^ r3
        r1 = n1 ^ r3
        r0 = n2 ^ r1
        final_s[r] = (
            state[r0] * operator[l, 0]
            + state[r1] * operator[l, 1]
            + state[r2] * operator[l, 2]
            + state[r3] * operator[l, 3]
        )
    return final_s
