import numpy as np

RECTANGLE_MODES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
    (3, 3),
    (2, 4),
    (4, 2),
    (1, 5),
    (5, 1),
    (3, 4),
]
# def rectangle_modal_factors(aspect, n_modes=16):
#     candidate_modes = [
#         (m, n)
#         for m in range(1, n_modes + 1)
#         for n in range(1, n_modes + 1)
#     ]

#     candidate_factors = np.asarray(
#         [
#             np.pi**2 * (
#                 m**2 / aspect
#                 + aspect * n**2
#             )
#             for m, n in candidate_modes
#         ],
#         dtype=np.float64,
#     )

#     order = np.argsort(candidate_factors)[:n_modes]

#     modes = [candidate_modes[i] for i in order]
#     modal_factors = candidate_factors[order]

#     return modes, modal_factors

def rectangle_modal_factors(aspect):
    factors = []

    for m, n in RECTANGLE_MODES:
        mu = np.pi**2 * (
            m**2 / aspect
            + aspect * n**2
        )

        factors.append(mu)

    return (
        RECTANGLE_MODES,
        np.asarray(factors, dtype=np.float32),
    )

# def rectangle_modal_gains(modes, x, y):
#     gains = [
#         np.sin(m * np.pi * x)
#         * np.sin(n * np.pi * y)
#         for m, n in modes
#     ]

#     return np.asarray(gains, dtype=np.float32)

def rectangle_modal_gains(modes, x, y):
    return np.asarray(
        [
            np.sin(m * np.pi * x)
            * np.sin(n * np.pi * y)
            for m, n in modes
        ],
        dtype=np.float32,
    )

if __name__ == "__main__":
    modes, factors = rectangle_modal_factors(1.0)

    gains = rectangle_modal_gains(
        modes,
        x=0.5,
        y=0.5,
    )

    print(modes)
    print(factors)
    print(gains)

# if __name__ == "__main__":
#     modes, factors = rectangle_modal_factors(1.0)

#     print(modes)
#     print(factors)
