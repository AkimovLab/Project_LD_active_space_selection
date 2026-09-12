"""Local-diabatization propagation and fixed-active-space diagnostics.

The routines in this module are independent of how energies and time overlaps
were generated.  Model trajectories and atomistic trajectories therefore use
the same support tests and unitary polar-LD propagator.
"""

from __future__ import annotations

import numpy as np


def polar_factor(overlap):
    """Return the closest unitary matrix to a square time-overlap matrix.

    Parameters
    ----------
    overlap : array_like, shape (N, N)
        Time-overlap block from geometry ``n`` to geometry ``n+1``.

    Returns
    -------
    numpy.ndarray, shape (N, N)
        The unitary polar factor ``U @ Vh`` of the singular-value
        decomposition ``overlap = U @ diag(sigma) @ Vh``.  All singular values
        are set to one; consequently this operation must be used only after
        the selected block has passed the numerical-support test.
    """
    left, _, right_h = np.linalg.svd(overlap, full_matrices=False)
    return left @ right_h


def symmetric_ld_step(transport, coefficients, energy_old, energy_new, dt):
    """Apply one symmetric local-diabatization propagation step.

    Parameters
    ----------
    transport : array_like, shape (N, N)
        Map from the old adiabatic basis to the new basis.  Standard production
        LD supplies the unitary polar factor of the active overlap block.
    coefficients : array_like, shape (N,) or (N, M)
        Adiabatic amplitudes at the beginning of the time-step interval.  A
        matrix propagates ``M`` initial conditions simultaneously.
    energy_old, energy_new : array_like, shape (N,)
        Adiabatic energies at the two endpoint geometries.
    dt : float
        Time-step interval in atomic units.

    Returns
    -------
    numpy.ndarray
        Coefficients at the new geometry after half an energy phase, basis
        transport by ``transport.conj().T``, and the second half phase.
    """
    coefficients = np.asarray(coefficients, dtype=complex)
    vector_input = coefficients.ndim == 1
    if vector_input:
        coefficients = coefficients[:, None]
    phase_old = np.exp(-0.5j * np.asarray(energy_old) * dt)[:, None]
    phase_new = np.exp(-0.5j * np.asarray(energy_new) * dt)[:, None]
    propagated = phase_new * (
        np.asarray(transport).conj().T @ (phase_old * coefficients)
    )
    return propagated[:, 0] if vector_input else propagated


def clean_time_overlap(
    overlap, relative_tolerance=1.0e-8, absolute_tolerance=1.0e-12
):
    """Measure the numerical support of one time-overlap matrix.

    Parameters
    ----------
    overlap : array_like, shape (N, N)
        Raw square time-overlap matrix.
    relative_tolerance : float, optional
        Relative noise threshold applied to the largest singular value.
    absolute_tolerance : float, optional
        Absolute lower bound for the singular-value threshold.

    Returns
    -------
    dict
        ``singular_values`` contains the descending spectrum, ``cutoff`` is
        ``max(absolute_tolerance, relative_tolerance*sigma_max)``, and ``rank``
        counts singular values strictly above that cutoff.  No matrix is
        modified by this diagnostic cleanup.
    """
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    cutoff = max(
        absolute_tolerance, relative_tolerance * singular_values[0]
    )
    return {
        "singular_values": singular_values,
        "cutoff": cutoff,
        "rank": int(np.count_nonzero(singular_values > cutoff)),
    }


def infer_constant_active_space(
    raw_overlaps, relative_tolerance=1.0e-3, absolute_tolerance=1.0e-12
):
    """Infer the trajectory-wide upper bound on a reliable fixed dimension.

    Parameters
    ----------
    raw_overlaps : array_like, shape (n_interval, N, N)
        Consecutive-geometry overlaps in the original computed state space.
    relative_tolerance, absolute_tolerance : float, optional
        Noise thresholds passed to :func:`clean_time_overlap`.

    Returns
    -------
    dict
        Per-interval singular spectra, cutoffs, reliable ranks, and
        ``constant_rank``, the minimum reliable rank over the trajectory.

    Notes
    -----
    ``constant_rank`` is only an upper bound on candidate dimensions.  It does
    not identify which fixed state labels form a supported principal block;
    candidate blocks must be tested with :func:`diagnose_fixed_active_spaces`.
    """
    raw_overlaps = np.asarray(raw_overlaps, dtype=complex)
    if raw_overlaps.ndim != 3 or raw_overlaps.shape[1] != raw_overlaps.shape[2]:
        raise ValueError("raw_overlaps must have shape (n_interval, N, N)")
    if len(raw_overlaps) == 0:
        raise ValueError("At least one time overlap is required")
    cleaned = [
        clean_time_overlap(item, relative_tolerance, absolute_tolerance)
        for item in raw_overlaps
    ]
    reliable_ranks = np.asarray([item["rank"] for item in cleaned])
    constant_rank = int(reliable_ranks.min())
    if constant_rank < 1:
        raise ValueError("No nonzero constant reliable space exists")
    return {
        "singular_values": np.asarray([
            item["singular_values"] for item in cleaned
        ]),
        "step_reliable_rank": reliable_ranks,
        "step_cutoff": np.asarray([item["cutoff"] for item in cleaned]),
        "constant_rank": constant_rank,
    }


def diagnose_fixed_active_spaces(
    raw_overlaps,
    active_sizes,
    relative_tolerance=1.0e-3,
    absolute_tolerance=1.0e-12,
):
    """Screen nested leading fixed blocks over an entire trajectory.

    Parameters
    ----------
    raw_overlaps : array_like, shape (n_interval, N, N)
        Raw time-overlap matrices in a consistently ordered adiabatic basis.
    active_sizes : iterable of int
        Dimensions of leading principal blocks to test.
    relative_tolerance, absolute_tolerance : float, optional
        Noise thresholds used independently for every block and interval.

    Returns
    -------
    dict
        Dictionary keyed by active-space size.  Each entry provides its
        singular spectra, minimum singular values, cutoffs, effective ranks,
        rank fractions, and trajectory-wide Boolean ``accepted`` decision.

    Notes
    -----
    A block is accepted only when every one of its singular values is above
    the noise cutoff at every time-step interval.  The function selects a
    space; it never inserts a rank-deficient partial isometry into propagation.
    """
    raw_overlaps = np.asarray(raw_overlaps, dtype=complex)
    if raw_overlaps.ndim != 3 or raw_overlaps.shape[1] != raw_overlaps.shape[2]:
        raise ValueError("raw_overlaps must have shape (n_interval, N, N)")
    nstates = raw_overlaps.shape[1]
    diagnostics = {}
    for n_active in active_sizes:
        if not 1 <= n_active <= nstates:
            raise ValueError("Every active size must lie between 1 and N")
        spectra = np.asarray([
            np.linalg.svd(item[:n_active, :n_active], compute_uv=False)
            for item in raw_overlaps
        ])
        cutoffs = np.maximum(
            absolute_tolerance, relative_tolerance * spectra[:, 0]
        )
        effective_rank = np.sum(spectra > cutoffs[:, None], axis=1)
        diagnostics[n_active] = {
            "singular_values": spectra,
            "sigma_min": spectra[:, -1],
            "effective_rank": effective_rank,
            "rank_fraction": effective_rank / n_active,
            "accepted": bool(np.all(effective_rank == n_active)),
            "cutoff": cutoffs,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        }
    return diagnostics


def active_boundary_energy_gaps(adiabatic_energies, active_sizes):
    """Summarize energy gaps at proposed upper active-space boundaries.

    Parameters
    ----------
    adiabatic_energies : array_like, shape (n_geometry, N)
        Energy-ordered adiabatic energies from data physically consistent with
        the corresponding overlaps.
    active_sizes : iterable of int
        Candidate sizes smaller than ``N``.  Size ``N_A`` places a boundary
        between states ``N_A-1`` and ``N_A``.

    Returns
    -------
    dict
        Candidate sizes and the minimum, fifth-percentile, and median adjacent
        energy gap over the trajectory.
    """
    energies = np.asarray(adiabatic_energies, dtype=float)
    sizes = np.asarray(active_sizes, dtype=int)
    if np.any(sizes < 1) or np.any(sizes >= energies.shape[1]):
        raise ValueError("Boundary sizes must satisfy 1 <= N_A < N")
    selected = np.diff(energies, axis=1)[:, sizes - 1]
    return {
        "active_sizes": sizes,
        "minimum": np.min(selected, axis=0),
        "fifth_percentile": np.quantile(selected, 0.05, axis=0),
        "median": np.median(selected, axis=0),
    }


def simulate_hamiltonian_active_space(
    hamiltonian,
    times,
    nstates,
    n_active,
    initial_state,
    reported_states=None,
    effective_rank_threshold=1.0e-3,
    unitarize_overlap=False,
    propagation_time_step=None,
):
    """Diagonalize a model Hamiltonian and propagate one active space.

    Parameters
    ----------
    hamiltonian : callable
        Function returning the full square Hamiltonian at one time.
    times : array_like
        Uniform model-coordinate grid used to evaluate the Hamiltonian.  It
        may be expressed in atomic units or femtoseconds.
    nstates : int
        Full model dimension.
    n_active : int
        Leading energy-ordered states retained in propagation.
    initial_state : int
        Initially occupied adiabatic-state index.
    reported_states : iterable of int, optional
        State populations to return; defaults to all active states.
    effective_rank_threshold : float, optional
        Relative threshold used for the diagnostic effective rank.
    unitarize_overlap : bool, optional
        If true, use the polar factor.  If false, use the contracted active
        overlap directly to demonstrate norm loss under a bad truncation.
    propagation_time_step : float, optional
        Physical time-step interval in atomic units.  When omitted, the
        difference between consecutive ``times`` values is used.  Supply this
        explicitly when the Hamiltonian coordinate is expressed in fs.

    Returns
    -------
    dict
        Populations, norm, active overlaps, minimum singular values, and
        effective ranks along the trajectory.
    """
    times = np.asarray(times, dtype=float)
    if n_active > nstates:
        raise ValueError("The active space cannot exceed the full space")
    if not 0 <= initial_state < n_active:
        raise ValueError("The initial state must lie in the active space")
    if reported_states is None:
        reported_states = np.arange(n_active)
    reported_states = np.asarray(reported_states, dtype=int)
    if np.any((reported_states < 0) | (reported_states >= n_active)):
        raise ValueError("Every reported state must lie in the active space")

    dt = (
        times[1] - times[0]
        if propagation_time_step is None else propagation_time_step
    )
    energy_old, vectors_old = np.linalg.eigh(hamiltonian(times[0]))
    coefficients = np.zeros(n_active, dtype=complex)
    coefficients[initial_state] = 1.0
    reported = [np.abs(coefficients[reported_states]) ** 2]
    active = [np.abs(coefficients) ** 2]
    sigma_min = [1.0]
    overlaps = [np.eye(n_active)]
    effective_rank = [n_active]

    for time in times[1:]:
        energy_new, vectors_new = np.linalg.eigh(hamiltonian(time))
        overlap = (vectors_old.conj().T @ vectors_new)[:n_active, :n_active]
        singular_values = np.linalg.svd(overlap, compute_uv=False)
        transport = polar_factor(overlap) if unitarize_overlap else overlap
        coefficients = symmetric_ld_step(
            transport,
            coefficients,
            energy_old[:n_active],
            energy_new[:n_active],
            dt,
        )
        reported.append(np.abs(coefficients[reported_states]) ** 2)
        active.append(np.abs(coefficients) ** 2)
        sigma_min.append(singular_values[-1])
        overlaps.append(overlap)
        effective_rank.append(np.count_nonzero(
            singular_values >= effective_rank_threshold * singular_values[0]
        ))
        energy_old, vectors_old = energy_new, vectors_new

    active = np.asarray(active)
    return {
        "reported_population": np.asarray(reported),
        "reported_states": reported_states,
        "active_population": active,
        "active_norm": active.sum(axis=1),
        "sigma_min": np.asarray(sigma_min),
        "time_overlaps": np.asarray(overlaps),
        "effective_rank": np.asarray(effective_rank),
        "effective_rank_threshold": effective_rank_threshold,
        "n_active": n_active,
        "initial_state": initial_state,
    }


def simulate_precomputed_overlap_ld(
    overlaps,
    adiabatic_energies,
    time_step,
    n_active,
    initial_state,
    reported_states=None,
):
    """Run unitary polar LD from precomputed adiabatic time overlaps.

    Parameters
    ----------
    overlaps : array_like, shape (n_interval, N, N)
        Authoritative precomputed adiabatic overlaps.  Matrix ``n`` maps the
        basis at geometry ``n`` to that at geometry ``n+1``.
    adiabatic_energies : array_like, shape (n_interval + 1, N)
        Adiabatic energies at the endpoint geometries.
    time_step : float
        Time-step interval in atomic units.
    n_active : int
        Dimension of the leading fixed square block.
    initial_state : int
        Initially occupied state index inside the block.
    reported_states : iterable of int, optional
        Common state populations to retain; defaults to every active state.

    Returns
    -------
    dict
        Reported populations, total active-space norm, minimum raw-block
        singular value, active size, and initial state.

    Notes
    -----
    Only the real adiabatic energies enter the symmetric dynamical phase.
    Off-diagonal vibronic-Hamiltonian entries are not added, avoiding double
    counting nonadiabatic information already present in the overlaps.
    """
    overlaps = np.asarray(overlaps, dtype=complex)
    energies = np.asarray(adiabatic_energies, dtype=float)
    if energies.shape[0] != len(overlaps) + 1:
        raise ValueError("One more energy geometry than overlap is required")
    if n_active > overlaps.shape[1]:
        raise ValueError("Active space exceeds overlap dimension")
    if not 0 <= initial_state < n_active:
        raise ValueError("initial_state must lie in the active space")
    if reported_states is None:
        reported_states = np.arange(n_active)
    reported_states = np.asarray(reported_states, dtype=int)
    if np.any((reported_states < 0) | (reported_states >= n_active)):
        raise ValueError("Every reported state must lie in the active space")

    coefficients = np.zeros(n_active, dtype=complex)
    coefficients[initial_state] = 1.0
    populations = [np.abs(coefficients[reported_states]) ** 2]
    norms = [1.0]
    sigma_min = [1.0]
    for step, raw_overlap in enumerate(overlaps):
        active_overlap = raw_overlap[:n_active, :n_active]
        coefficients = symmetric_ld_step(
            polar_factor(active_overlap),
            coefficients,
            energies[step, :n_active],
            energies[step + 1, :n_active],
            time_step,
        )
        populations.append(np.abs(coefficients[reported_states]) ** 2)
        norms.append(np.vdot(coefficients, coefficients).real)
        sigma_min.append(np.linalg.svd(active_overlap, compute_uv=False)[-1])
    return {
        "reported_population": np.asarray(populations),
        "reported_states": reported_states,
        "active_norm": np.asarray(norms),
        "sigma_min": np.asarray(sigma_min),
        "n_active": n_active,
        "initial_state": initial_state,
    }


def compare_fixed_active_spaces(
    times,
    energies,
    exact_overlaps,
    observed_overlaps,
    active_sizes,
    initial_state,
    reported_states,
    time_step=None,
):
    """Compare unitary LD in observed blocks with an ideal full reference.

    Parameters
    ----------
    times : array_like
        Uniform geometry times in atomic units.
    energies : array_like, shape (n_geometry, N)
        Reference adiabatic energies.
    exact_overlaps, observed_overlaps : array_like
        Overlap arrays containing one identity-like entry at the first geometry
        followed by consecutive-geometry overlaps.  The exact array supplies
        the full-space reference; the observed array supplies candidate blocks.
    active_sizes : iterable of int
        Candidate dimensions.
    initial_state : int
        Common initial state.
    reported_states : iterable of int
        Common populations compared across all candidates.
    time_step : float, optional
        Physical time-step interval in atomic units.  Defaults to the spacing
        of ``times``; pass it explicitly when ``times`` is in femtoseconds.

    Returns
    -------
    dict
        Candidate ``results`` keyed by dimension and an ideal full ``reference``.
    """
    dt = (
        np.asarray(times)[1] - np.asarray(times)[0]
        if time_step is None else time_step
    )
    results = {
        size: simulate_precomputed_overlap_ld(
            observed_overlaps[1:], energies, dt, size, initial_state,
            reported_states,
        )
        for size in active_sizes
    }
    reference = simulate_precomputed_overlap_ld(
        exact_overlaps[1:], energies, dt, exact_overlaps.shape[1],
        initial_state, reported_states,
    )
    return {"results": results, "reference": reference}


def propagate_multiple_initial_states(
    overlaps, energies, time_step, n_active, initial_states
):
    """Propagate several initial states through one polar-LD active space.

    Parameters
    ----------
    overlaps : array_like, shape (n_interval, N, N)
        Precomputed adiabatic time overlaps.
    energies : array_like, shape (n_interval + 1, N)
        Adiabatic energies.
    time_step : float
        Time-step interval in atomic units.
    n_active : int
        Fixed leading-block dimension.
    initial_states : iterable of int
        Initial states smaller than ``n_active``.

    Returns
    -------
    dict
        For each initial state, populations of states ``0`` through that state.
        Restricting the observable set this way makes it common to every
        admissible candidate in a systematic size series.
    """
    valid_states = [state for state in initial_states if state < n_active]
    coefficients = np.zeros((n_active, len(valid_states)), dtype=complex)
    for column, state in enumerate(valid_states):
        coefficients[state, column] = 1.0
    populations = {
        state: [np.abs(coefficients[:state + 1, column]) ** 2]
        for column, state in enumerate(valid_states)
    }
    for step, overlap in enumerate(overlaps):
        coefficients = symmetric_ld_step(
            polar_factor(overlap[:n_active, :n_active]),
            coefficients,
            energies[step, :n_active],
            energies[step + 1, :n_active],
            time_step,
        )
        for column, state in enumerate(valid_states):
            populations[state].append(
                np.abs(coefficients[:state + 1, column]) ** 2
            )
    return {state: np.asarray(values) for state, values in populations.items()}


def systematic_active_size_convergence(
    overlaps,
    energies,
    time_step,
    active_sizes,
    initial_states,
    reference_overlaps=None,
    reference_size=None,
    relative_tolerance=1.0e-3,
):
    """Compute population convergence for systematic active-space sizes.

    Parameters
    ----------
    overlaps, energies, time_step : array_like, array_like, float
        Precomputed-overlap LD input.
    active_sizes : iterable of int
        Candidate dimensions tested in the requested order.
    initial_states : iterable of int
        Separate fixed initial states used to expose boundary sensitivity.
    reference_overlaps : array_like, optional
        Alternative overlaps for the comparison calculation, such as the exact
        model overlaps when candidates use corrupted observations.
    reference_size : int, optional
        Comparison dimension.  Defaults to the full energy dimension.
    relative_tolerance : float, optional
        Support-screening threshold stored with the convergence results.

    Returns
    -------
    dict
        Candidate sizes and trajectory population RMSE for each initial state,
        plus numerical-support diagnostics for every candidate size.
    """
    active_sizes = list(active_sizes)
    initial_states = list(initial_states)
    if reference_overlaps is None:
        reference_overlaps = overlaps
    if reference_size is None:
        reference_size = np.asarray(energies).shape[1]
    if any(state >= reference_size for state in initial_states):
        raise ValueError("Every initial state must lie in the reference space")
    reference = propagate_multiple_initial_states(
        reference_overlaps, energies, time_step, reference_size, initial_states
    )
    rmse = {state: [] for state in initial_states}
    sizes_by_state = {state: [] for state in initial_states}
    for size in active_sizes:
        valid = [state for state in initial_states if state < size]
        if not valid:
            continue
        current = propagate_multiple_initial_states(
            overlaps, energies, time_step, size, valid
        )
        for state in valid:
            sizes_by_state[state].append(size)
            rmse[state].append(np.sqrt(np.mean(
                (current[state] - reference[state]) ** 2
            )))
    return {
        "active_sizes": {
            state: np.asarray(values) for state, values in sizes_by_state.items()
        },
        "rmse": {state: np.asarray(values) for state, values in rmse.items()},
        "diagnostics": diagnose_fixed_active_spaces(
            overlaps, active_sizes, relative_tolerance=relative_tolerance
        ),
    }


def population_rmse(first, second):
    """Return RMS population difference between two result dictionaries.

    Parameters
    ----------
    first, second : dict
        Results containing equal-shaped ``reported_population`` arrays.

    Returns
    -------
    float
        Root-mean-square difference over all times and reported states.
    """
    difference = first["reported_population"] - second["reported_population"]
    return float(np.sqrt(np.mean(difference**2)))


def cumulative_population_rmse(result, reference):
    """Return common-state RMSE accumulated from the trajectory start.

    Parameters
    ----------
    result, reference : dict
        Propagation results with equal-shaped ``reported_population`` arrays.

    Returns
    -------
    numpy.ndarray
        Running RMSE, with element ``m`` accumulated through geometry ``m``.
    """
    difference = result["reported_population"] - reference["reported_population"]
    squared_error = np.sum(difference**2, axis=1)
    count = np.arange(1, len(difference) + 1) * difference.shape[1]
    return np.sqrt(np.cumsum(squared_error) / count)


def convergence_diagnostics(results, active_sizes):
    """Summarize convergence and overlap diagnostics for nested model runs.

    Parameters
    ----------
    results : dict
        Propagation results keyed by active-space dimension.
    active_sizes : sequence of int
        Ordered candidate sizes; the last is used as the internal reference.

    Returns
    -------
    dict
        RMSE versus the last size, RMSE versus the next size, minimum singular
        value, and minimum effective-rank fraction for each candidate.
    """
    reference = results[active_sizes[-1]]
    reference_rmse, adjacent_rmse = [], []
    minimum_sigma, minimum_rank_fraction = [], []
    for index, size in enumerate(active_sizes):
        current = results[size]
        reference_rmse.append(population_rmse(current, reference))
        adjacent_rmse.append(
            population_rmse(current, results[active_sizes[index + 1]])
            if index + 1 < len(active_sizes) else np.nan
        )
        minimum_sigma.append(np.min(current["sigma_min"]))
        minimum_rank_fraction.append(np.min(
            current["effective_rank"] / size
        ))
    return {
        "reference_rmse": np.asarray(reference_rmse),
        "adjacent_rmse": np.asarray(adjacent_rmse),
        "minimum_sigma": np.asarray(minimum_sigma),
        "minimum_effective_rank_fraction": np.asarray(minimum_rank_fraction),
    }
