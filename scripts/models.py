"""Deterministic Hamiltonian models and controlled overlap stress tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


FEMTOSECOND_TO_ATOMIC_TIME = 41.3413745758


SIMPLE_MODEL_PARAMS = {
    "model": "three_state",
    "nstates": 3,
    "time_start": 0.0,
    "time_stop": 10.0,
    "time_step": 0.05,
    "time_unit": "atomic",
    "energy_amplitude": 0.005,
    "coupling_01": 0.002,
    "overlap_filter": "none",
}


DENSE_MODEL_PARAMS = {
    "model": "dense_manifold",
    "nstates": 30,
    "time_start": 0.0,
    "time_stop": 120.0,
    "time_step": 0.5,
    "time_unit": "fs",
    "band_boundaries": [6, 10, 14, 20, 24],
    "first_energy": -0.055,
    "intraband_spacing": 0.0012,
    "interband_spacing": 0.0090,
    "energy_amplitude_offset": 0.0032,
    "energy_amplitude_scale": 0.0010,
    "energy_amplitude_frequency": 0.73,
    "energy_amplitude_phase": 0.20,
    "frequency_offset": 0.365,
    "frequency_scale": 0.18,
    "frequency_index_scale": 0.41,
    "frequency_phase": 0.30,
    "phase_linear": 0.91,
    "phase_quadratic": 0.07,
    "drift_scale": 0.000075,
    "drift_index_scale": 1.13,
    "drift_phase": 0.40,
    "drift_center": 20.0,
    "maximum_coupling_separation": 8,
    "coupling_scale": 0.0025,
    "coupling_decay": 3.0,
    "coupling_modulation": 0.30,
    "coupling_modulation_frequency": 0.13,
    "overlap_filter": "none",
    "reported_count": 6,
    "filter_mixing_angle": 0.08,
    "filter_noise_scale": 1.0e-4,
    "filter_noise_phase": 0.0,
    "hard_windows": [
        (8.0, 4, 0.0),
        (16.0, 6, 0.0),
        (29.0, 12, 0.0),
        (np.inf, 8, 1.0e-15),
    ],
    "ill_supported_count": 10,
    "ill_tail_start": 1.0e-2,
    "ill_tail_stop": 1.0e-14,
    "ill_oscillation_amplitude": 0.7,
    "ill_oscillation_frequency": 0.23,
}


def model_parameters(name="dense_manifold", **overrides):
    """Return an independent parameter dictionary for a named model.

    Parameters
    ----------
    name : {"three_state", "dense_manifold"}, optional
        Base model whose documented defaults are copied.
    **overrides
        Parameter values replacing defaults.  In particular,
        ``overlap_filter`` may be ``"none"``, ``"hard_collapse"``, or
        ``"ill_conditioned"`` for the dense model.

    Returns
    -------
    dict
        Mutable deep copy suitable for one calculation.  Editing it does not
        alter the module defaults.
    """
    if name == "three_state":
        params = deepcopy(SIMPLE_MODEL_PARAMS)
    elif name == "dense_manifold":
        params = deepcopy(DENSE_MODEL_PARAMS)
    else:
        raise ValueError(f"Unknown model: {name}")
    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"Unknown model parameter(s): {sorted(unknown)}")
    params.update(overrides)
    return params


def model_time_grid(params):
    """Construct the inclusive uniform time grid specified by ``params``.

    Parameters
    ----------
    params : dict
        Model dictionary containing ``time_start``, ``time_stop``, and
        ``time_step`` in the unit specified by ``time_unit``.

    Returns
    -------
    numpy.ndarray
        Uniform geometry times including the requested endpoint.  The values
        are model coordinates: atomic units for the three-state example and
        femtoseconds for the dense model.
    """
    start = params["time_start"]
    stop = params["time_stop"]
    step = params["time_step"]
    return np.arange(start, stop + 0.5 * step, step)


def three_state_hamiltonian(time, params):
    """Evaluate the diabatic three-state truncation model.

    Parameters
    ----------
    time : float or array_like
        Nuclear-trajectory time in atomic units.
    params : dict
        Three-state model parameters.  ``energy_amplitude`` controls the
        opposite cosine energies of states 1 and 2; ``coupling_01`` is the only
        nonzero off-diagonal coupling.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)`` for scalar input or ``(n_time, 3, 3)`` for an array.
    """
    values = np.atleast_1d(np.asarray(time, dtype=float))
    energy = params["energy_amplitude"] * np.cos(values)
    matrices = np.zeros((len(values), 3, 3), dtype=float)
    matrices[:, 1, 1] = energy
    matrices[:, 2, 2] = -energy
    matrices[:, 0, 1] = params["coupling_01"]
    matrices[:, 1, 0] = params["coupling_01"]
    return matrices[0] if np.ndim(time) == 0 else matrices


def dense_model_arrays(params):
    """Precompute all deterministic arrays defining the dense Hamiltonian.

    Parameters
    ----------
    params : dict
        Dense-model parameter dictionary returned by :func:`model_parameters`.
        It specifies band spacings, oscillatory energies, linear drifts, and
        finite-range time-dependent diabatic couplings.

    Returns
    -------
    dict
        Per-state base energies, amplitudes, frequencies, phases, and drifts,
        plus symmetric coupling and coupling-phase matrices.
    """
    nstates = params["nstates"]
    index = np.arange(nstates, dtype=float)
    base = np.empty(nstates)
    base[0] = params["first_energy"]
    boundaries = set(params["band_boundaries"])
    for state in range(1, nstates):
        base[state] = base[state - 1] + (
            params["interband_spacing"] if state in boundaries
            else params["intraband_spacing"]
        )
    amplitude = (
        params["energy_amplitude_offset"]
        + params["energy_amplitude_scale"] * np.sin(
            params["energy_amplitude_frequency"] * index
            + params["energy_amplitude_phase"]
        )
    )
    frequency = (
        params["frequency_offset"]
        + params["frequency_scale"] * np.cos(
            params["frequency_index_scale"] * index
            + params["frequency_phase"]
        )
    )
    phase = np.mod(
        params["phase_linear"] * index
        + params["phase_quadratic"] * index**2,
        2.0 * np.pi,
    )
    drift = params["drift_scale"] * np.sin(
        params["drift_index_scale"] * index + params["drift_phase"]
    )
    coupling = np.zeros((nstates, nstates))
    coupling_phase = np.zeros_like(coupling)
    maximum = params["maximum_coupling_separation"]
    for first in range(nstates):
        for second in range(first + 1, min(nstates, first + maximum + 1)):
            separation = second - first
            value = (
                params["coupling_scale"]
                * np.exp(-(separation - 1) / params["coupling_decay"])
                * (
                    0.65 * np.sin(0.37 * (first + 1) * (second + 1))
                    + 0.35 * np.cos(0.29 * (first + second + 2))
                )
            )
            coupling[first, second] = coupling[second, first] = value
            angle = np.mod(
                0.53 * (first + 1) + 0.71 * (second + 1)
                + 0.11 * (first + 1) * (second + 1),
                2.0 * np.pi,
            )
            coupling_phase[first, second] = angle
            coupling_phase[second, first] = angle
    return {
        "base": base,
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "drift": drift,
        "coupling": coupling,
        "coupling_phase": coupling_phase,
    }


def make_dense_hamiltonian(params):
    """Create an efficient callable for the deterministic dense manifold.

    Parameters
    ----------
    params : dict
        Dense-model parameters.  Arrays that do not depend on time are built
        once when this factory is called.

    Returns
    -------
    callable
        Function ``hamiltonian(time)`` returning one real symmetric diabatic
        Hamiltonian of dimension ``params['nstates']``.
    """
    arrays = dense_model_arrays(params)

    def hamiltonian(time):
        """Evaluate the configured dense diabatic Hamiltonian at one time."""
        diagonal = (
            arrays["base"]
            + arrays["amplitude"] * np.sin(
                arrays["frequency"] * time + arrays["phase"]
            )
            + arrays["drift"] * (time - params["drift_center"])
        )
        modulation = 1.0 + params["coupling_modulation"] * np.sin(
            params["coupling_modulation_frequency"] * time
            + arrays["coupling_phase"]
        )
        return np.diag(diagonal) + arrays["coupling"] * modulation

    return hamiltonian


def make_hamiltonian(params):
    """Return the Hamiltonian callable selected by a model dictionary.

    Parameters
    ----------
    params : dict
        Dictionary whose ``model`` key is ``"three_state"`` or
        ``"dense_manifold"``.

    Returns
    -------
    callable
        Scalar-time Hamiltonian evaluator.
    """
    if params["model"] == "three_state":
        return lambda time: three_state_hamiltonian(time, params)
    if params["model"] == "dense_manifold":
        return make_dense_hamiltonian(params)
    raise ValueError(f"Unknown model: {params['model']}")


def givens_rotation(dimension, first, second, angle):
    """Construct a real Givens rotation in one coordinate plane.

    Parameters
    ----------
    dimension : int
        Dimension of the full identity matrix.
    first, second : int
        Distinct coordinate indices defining the rotated plane.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Identity except for rows and columns ``first, second``, whose block is
        ``[[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]``.
    """
    if first == second or not (0 <= first < dimension and 0 <= second < dimension):
        raise ValueError("Givens indices must be distinct and inside the matrix")
    rotation = np.eye(dimension)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation[first, first] = cosine
    rotation[second, second] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    return rotation


def reliability_basis(params):
    """Build the fixed basis in which synthetic reliability is diagonal.

    Parameters
    ----------
    params : dict
        Dense parameters containing ``nstates``, ``reported_count``, and
        ``filter_mixing_angle``.  Each low reported direction is mixed with a
        distinct direction at the upper end of the state list.

    Returns
    -------
    numpy.ndarray
        Orthogonal product of disjoint Givens rotations.
    """
    dimension = params["nstates"]
    count = params["reported_count"]
    basis = np.eye(dimension)
    for state in range(count):
        basis = basis @ givens_rotation(
            dimension, state, dimension - count + state,
            params["filter_mixing_angle"],
        )
    return basis


def reliability_history(times, params):
    """Construct deterministic singular-direction reliability factors.

    Parameters
    ----------
    times : array_like
        Geometry times in atomic units.
    params : dict
        Dense-model dictionary. ``overlap_filter`` chooses ``"none"``,
        ``"hard_collapse"``, or ``"ill_conditioned"``.  Hard collapse uses
        ``hard_windows`` tuples ``(upper_time, tail_count, factor)``.  The
        ill-conditioned filter uses the documented log-spaced tail parameters.

    Returns
    -------
    numpy.ndarray, shape (n_geometry, N)
        Multiplicative reliability factors before rotation to the observation
        basis.  These factors are synthetic and are not physical state weights.
    """
    times = np.asarray(times)
    nstates = params["nstates"]
    filter_name = params.get("overlap_filter", "none")
    history = np.ones((len(times), nstates))
    if filter_name == "none":
        return history
    if filter_name == "hard_collapse":
        for row, time in enumerate(times):
            for upper_time, count, factor in params["hard_windows"]:
                if time < upper_time:
                    history[row, -count:] = factor
                    break
        return history
    if filter_name == "ill_conditioned":
        supported = params["ill_supported_count"]
        tail = np.logspace(
            np.log10(params["ill_tail_start"]),
            np.log10(params["ill_tail_stop"]),
            nstates - supported,
        )
        tail_index = np.arange(len(tail))
        for row, time in enumerate(times):
            oscillation = 10.0 ** (
                params["ill_oscillation_amplitude"] * np.sin(
                    params["ill_oscillation_frequency"] * time + tail_index
                )
            )
            history[row, supported:] = tail * oscillation
        return history
    raise ValueError(f"Unknown overlap_filter: {filter_name}")


def deterministic_observed_overlaps(
    exact_overlaps,
    reliability,
    noise_scale=0.0,
    reliability_basis_matrix=None,
    noise_phase=0.0,
):
    """Apply a reliability operator and deterministic perturbation to overlaps.

    Parameters
    ----------
    exact_overlaps : array_like, shape (n_geometry, N, N)
        Exact model overlaps, including an identity at the first geometry.
    reliability : array_like, shape (n_geometry, N)
        Per-geometry singular-direction factors.
    noise_scale : float, optional
        Amplitude of the explicit deterministic perturbation before division
        by ``sqrt(N)``.
    reliability_basis_matrix : array_like, optional
        Orthogonal basis rotating the diagonal reliability operator.
    noise_phase : float, optional
        Phase changing the deterministic perturbation without random draws.

    Returns
    -------
    numpy.ndarray
        Synthetic observed overlaps.  They intentionally need not be
        consistent with the Hamiltonian used to create the exact overlaps.
    """
    exact_overlaps = np.asarray(exact_overlaps, dtype=complex)
    reliability = np.asarray(reliability, dtype=float)
    nstates = exact_overlaps.shape[1]
    row = np.arange(nstates)[:, None] + 1.0
    column = np.arange(nstates)[None, :] + 1.0
    observed = []
    for step, (exact, factors) in enumerate(zip(exact_overlaps, reliability)):
        operator = np.diag(factors)
        if reliability_basis_matrix is not None:
            operator = (
                reliability_basis_matrix @ operator
                @ reliability_basis_matrix.T
            )
        matrix = exact @ operator
        if noise_scale:
            noise = (
                np.sin(
                    0.37 * row * column + 0.17 * step * (row + column)
                    + 0.11 * step**2 + noise_phase
                )
                + np.cos(
                    0.23 * (row + step) * (column + 2.0 * step)
                    + 0.41 * row + 0.13 * column - 0.7 * noise_phase
                )
            )
            matrix = matrix + noise_scale * noise / np.sqrt(nstates)
        observed.append(matrix)
    return np.asarray(observed)


def build_model_trajectory(params):
    """Generate Hamiltonians, energies, exact overlaps, and optional stress data.

    Parameters
    ----------
    params : dict
        Complete model dictionary.  The ``overlap_filter`` keyword controls the
        observed overlaps: ``"none"`` returns exact overlaps,
        ``"hard_collapse"`` removes prescribed tail directions, and
        ``"ill_conditioned"`` applies a long singular-value tail.  Filtered
        overlaps are deliberately changed independently of the Hamiltonian.

    Returns
    -------
    dict
        Parameters, model-coordinate time grid, atomic-unit propagation
        interval, Hamiltonian callable, adiabatic energies,
        eigenvectors, exact overlaps, observed overlaps, reliability history,
        and reliability basis.  Every overlap array includes an identity-like
        entry at index zero; propagation uses entries ``1:``.
    """
    params = deepcopy(params)
    times = model_time_grid(params)
    if params["time_unit"] == "atomic":
        time_step_atomic = params["time_step"]
    elif params["time_unit"] == "fs":
        time_step_atomic = params["time_step"] * FEMTOSECOND_TO_ATOMIC_TIME
    else:
        raise ValueError("time_unit must be 'atomic' or 'fs'")
    hamiltonian = make_hamiltonian(params)
    energies, vectors = zip(*[
        np.linalg.eigh(hamiltonian(time)) for time in times
    ])
    energies = np.asarray(energies)
    vectors = np.asarray(vectors)
    exact_overlaps = [np.eye(params["nstates"])]
    exact_overlaps.extend(
        vectors[index].conj().T @ vectors[index + 1]
        for index in range(len(times) - 1)
    )
    exact_overlaps = np.asarray(exact_overlaps)
    factors = reliability_history(times, params)
    if params.get("overlap_filter", "none") == "none":
        basis = np.eye(params["nstates"])
        observed = exact_overlaps.copy()
    else:
        if params["model"] != "dense_manifold":
            raise ValueError("Overlap stress filters require the dense model")
        basis = reliability_basis(params)
        observed = deterministic_observed_overlaps(
            exact_overlaps,
            factors,
            noise_scale=params["filter_noise_scale"],
            reliability_basis_matrix=basis,
            noise_phase=params["filter_noise_phase"],
        )
    return {
        "params": params,
        "times": times,
        "time_unit": params["time_unit"],
        "time_step_atomic": time_step_atomic,
        "hamiltonian": hamiltonian,
        "energies": energies,
        "eigenvectors": vectors,
        "exact_overlaps": exact_overlaps,
        "observed_overlaps": observed,
        "reliability_history": factors,
        "reliability_basis": basis,
    }
