"""Loading and preprocessing of atomistic adiabatic LD trajectories."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.sparse import load_npz


FEMTOSECOND_TO_ATOMIC_TIME = 41.3413745758


_FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_COMPLEX_TUPLE_PATTERN = re.compile(
    rf"\(\s*({_FLOAT_PATTERN})\s*,\s*({_FLOAT_PATTERN})\s*\)"
)


def _indexed_sparse_files(directory, pattern):
    """Map trajectory indices to sparse files matching one naming pattern.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing the sparse NPZ trajectory.
    pattern : str
        Glob pattern whose filenames end in ``_<index>_<part>.npz``.

    Returns
    -------
    dict
        Integer trajectory indices mapped to matching paths.
    """
    return {
        int(path.name.split("_")[-2]): path
        for path in directory.glob(pattern)
    }


def _indexed_text_files(directory, pattern):
    """Map integer suffixes to text files matching one glob pattern.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing files such as ``ham_adi_1000.txt``.
    pattern : str
        Glob pattern selecting the desired matrix series.

    Returns
    -------
    dict
        Integer index after the final underscore mapped to each file path.
    """
    directory = Path(directory)
    indexed = {}
    for path in directory.glob(pattern):
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Cannot extract a trailing integer index from {path.name}"
            ) from error
        indexed[index] = path
    return indexed


def load_complex_tuple_text_matrix(path):
    """Read a dense matrix whose entries are ``(real,imaginary)`` tuples.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file containing one matrix row per line. Whitespace around tuple
        components and scientific notation are accepted.

    Returns
    -------
    numpy.ndarray
        Two-dimensional complex-valued matrix.

    Raises
    ------
    ValueError
        If rows have inconsistent lengths, contain unparsed non-whitespace
        text, or do not form a square matrix.
    """
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        matches = list(_COMPLEX_TUPLE_PATTERN.finditer(line))
        residue = _COMPLEX_TUPLE_PATTERN.sub("", line).strip()
        if not matches or residue:
            raise ValueError(
                f"Malformed complex-tuple row in {path} at line {line_number}"
            )
        rows.append([
            float(match.group(1)) + 1j * float(match.group(2))
            for match in matches
        ])
    if not rows:
        raise ValueError(f"No matrix entries found in {path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent row lengths in {path}: {sorted(widths)}")
    matrix = np.asarray(rows, dtype=complex)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix in {path}, got {matrix.shape}")
    return matrix


def load_real_text_matrix(path):
    """Read and validate one whitespace-delimited dense real matrix.

    Parameters
    ----------
    path : str or pathlib.Path
        Text matrix readable by :func:`numpy.loadtxt`.

    Returns
    -------
    numpy.ndarray
        Two-dimensional square floating-point matrix.
    """
    path = Path(path)
    matrix = np.loadtxt(path, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix in {path}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite value found in {path}")
    return matrix


def _longest_contiguous_geometry_run(hamiltonian_ids, overlap_ids):
    """Find the longest run containing every required H/S/H transition.

    Parameters
    ----------
    hamiltonian_ids : set of int
        Indices for which a complete Hamiltonian matrix exists.
    overlap_ids : iterable of int
        Indices with a precomputed overlap from ``i`` to ``i+1``.

    Returns
    -------
    list of int
        Consecutive geometry indices.  A run of ``M`` geometries has ``M-1``
        valid time-step intervals.
    """
    transition_ids = sorted(
        index for index in overlap_ids
        if index in hamiltonian_ids and index + 1 in hamiltonian_ids
    )
    if not transition_ids:
        raise ValueError(
            "No complete Hamiltonian_i/overlap_i/Hamiltonian_(i+1) "
            "transition found"
        )
    runs = []
    start = previous = transition_ids[0]
    for index in transition_ids[1:]:
        if index != previous + 1:
            runs.append(list(range(start, previous + 2)))
            start = index
        previous = index
    runs.append(list(range(start, previous + 2)))
    return max(runs, key=len)


def load_sparse_atomistic_trajectory(directory):
    """Load adiabatic-basis ``Hvib_ci`` and precomputed ``St_ci`` matrices.

    Parameters
    ----------
    directory : str or pathlib.Path
        Folder containing ``Hvib_ci_<i>_re.npz``,
        ``Hvib_ci_<i>_im.npz``, and ``St_ci_<i>_re.npz`` files.

    Returns
    -------
    dict
        Longest contiguous geometry indices, complex vibronic Hamiltonians,
        their real diagonal adiabatic energies, authoritative precomputed time
        overlaps, and the resolved source directory.

    Notes
    -----
    ``St_ci_i`` is interpreted as the overlap from geometry ``i`` to ``i+1``.
    The vibronic Hamiltonians are already in the adiabatic electronic
    representation; they are not diabatic Hamiltonians.  The overlaps are read
    directly and are never reconstructed by diagonalizing ``Hvib_ci``.
    """
    directory = Path(directory).expanduser().resolve()
    real_hamiltonians = _indexed_sparse_files(
        directory, "Hvib_ci_*_re.npz"
    )
    imaginary_hamiltonians = _indexed_sparse_files(
        directory, "Hvib_ci_*_im.npz"
    )
    overlap_files = _indexed_sparse_files(directory, "St_ci_*_re.npz")
    hamiltonian_ids = set(real_hamiltonians) & set(imaginary_hamiltonians)
    if not hamiltonian_ids:
        raise FileNotFoundError(f"No Hvib_ci trajectory found in {directory}")
    indices = _longest_contiguous_geometry_run(
        hamiltonian_ids, overlap_files
    )
    hamiltonians = np.asarray([
        load_npz(real_hamiltonians[index]).toarray()
        + 1j * load_npz(imaginary_hamiltonians[index]).toarray()
        for index in indices
    ])
    overlaps = np.asarray([
        load_npz(overlap_files[index]).toarray()
        for index in indices[:-1]
    ], dtype=complex)
    return {
        "indices": np.asarray(indices),
        "hamiltonians": hamiltonians,
        "adiabatic_energies": np.real(np.diagonal(
            hamiltonians, axis1=1, axis2=2
        )),
        "overlaps": overlaps,
        "directory": directory,
    }


def load_text_atomistic_trajectory(
    directory,
    energy_subdirectory="energy",
    overlap_subdirectory="time_overlap",
    hamiltonian_pattern="ham_adi_*.txt",
    overlap_pattern="st_adi_*.txt",
):
    """Load tuple-encoded Hamiltonians and real time overlaps from text files.

    Parameters
    ----------
    directory : str or pathlib.Path
        Parent directory containing separate energy and overlap folders.
    energy_subdirectory, overlap_subdirectory : str, optional
        Relative folder names containing the two matrix series.
    hamiltonian_pattern, overlap_pattern : str, optional
        Glob patterns whose final underscore-delimited stem component is an
        integer trajectory index.

    Returns
    -------
    dict
        Longest contiguous geometry indices, complex adiabatic Hamiltonians,
        their real diagonal energies, real-valued precomputed time overlaps
        stored as complex arrays for LD, source paths, and unused overlap
        indices that lack a matching next Hamiltonian.

    Notes
    -----
    ``st_adi_i`` is interpreted as the overlap from geometry ``i`` to
    geometry ``i+1``.  The Hamiltonians are already in the adiabatic basis;
    no diagonalization is used to reconstruct the supplied overlaps.
    """
    directory = Path(directory).expanduser().resolve()
    energy_directory = directory / energy_subdirectory
    overlap_directory = directory / overlap_subdirectory
    hamiltonian_files = _indexed_text_files(
        energy_directory, hamiltonian_pattern
    )
    overlap_files = _indexed_text_files(overlap_directory, overlap_pattern)
    if not hamiltonian_files:
        raise FileNotFoundError(
            f"No Hamiltonian files matching {hamiltonian_pattern} in "
            f"{energy_directory}"
        )
    if not overlap_files:
        raise FileNotFoundError(
            f"No overlap files matching {overlap_pattern} in "
            f"{overlap_directory}"
        )
    indices = _longest_contiguous_geometry_run(
        set(hamiltonian_files), overlap_files
    )
    hamiltonians = np.asarray([
        load_complex_tuple_text_matrix(hamiltonian_files[index])
        for index in indices
    ])
    overlaps = np.asarray([
        load_real_text_matrix(overlap_files[index])
        for index in indices[:-1]
    ], dtype=complex)
    dimension = hamiltonians.shape[1]
    expected_hamiltonian_shape = (len(indices), dimension, dimension)
    expected_overlap_shape = (len(indices) - 1, dimension, dimension)
    if hamiltonians.shape != expected_hamiltonian_shape:
        raise ValueError(
            "Hamiltonian dimensions change along the text trajectory: "
            f"{hamiltonians.shape}"
        )
    if overlaps.shape != expected_overlap_shape:
        raise ValueError(
            "Hamiltonian and overlap dimensions are inconsistent: "
            f"{hamiltonians.shape} versus {overlaps.shape}"
        )
    used_overlap_ids = set(indices[:-1])
    return {
        "indices": np.asarray(indices),
        "hamiltonians": hamiltonians,
        "adiabatic_energies": np.real(np.diagonal(
            hamiltonians, axis1=1, axis2=2
        )),
        "overlaps": overlaps,
        "directory": directory,
        "energy_directory": energy_directory,
        "overlap_directory": overlap_directory,
        "unused_overlap_indices": np.asarray(sorted(
            set(overlap_files) - used_overlap_ids
        ), dtype=int),
    }


def prepare_atomistic_trajectory(directory, time_step_fs=1.0):
    """Load atomistic data and add propagation-ready time information.

    Parameters
    ----------
    directory : str or pathlib.Path
        Sparse trajectory directory accepted by
        :func:`load_sparse_atomistic_trajectory`.
    time_step_fs : float, optional
        Physical duration of one indexed time-step interval in femtoseconds.

    Returns
    -------
    dict
        Loaded atomistic arrays plus ``times_fs``, ``overlap_times_fs``, and
        ``time_step_atomic``.  The returned dictionary can be passed directly
        to active-space diagnostics and precomputed-overlap LD propagation.
    """
    if time_step_fs <= 0.0:
        raise ValueError("time_step_fs must be positive")
    data = load_sparse_atomistic_trajectory(directory)
    relative_indices = data["indices"] - data["indices"][0]
    data["times_fs"] = relative_indices.astype(float) * time_step_fs
    data["overlap_times_fs"] = data["times_fs"][:-1]
    data["time_step_fs"] = float(time_step_fs)
    data["time_step_atomic"] = (
        float(time_step_fs) * FEMTOSECOND_TO_ATOMIC_TIME
    )
    return data


def prepare_text_atomistic_trajectory(directory, time_step_fs=1.0, **kwargs):
    """Load a text trajectory and add propagation-ready time information.

    Parameters
    ----------
    directory : str or pathlib.Path
        Parent directory accepted by :func:`load_text_atomistic_trajectory`.
    time_step_fs : float, optional
        Physical duration of one indexed time-step interval in femtoseconds.
    **kwargs
        Optional subdirectory names and filename patterns forwarded to
        :func:`load_text_atomistic_trajectory`.

    Returns
    -------
    dict
        Loaded matrices plus geometry times, overlap-interval times, and the
        time-step interval converted to atomic units.
    """
    if time_step_fs <= 0.0:
        raise ValueError("time_step_fs must be positive")
    data = load_text_atomistic_trajectory(directory, **kwargs)
    relative_indices = data["indices"] - data["indices"][0]
    data["times_fs"] = relative_indices.astype(float) * time_step_fs
    data["overlap_times_fs"] = data["times_fs"][:-1]
    data["time_step_fs"] = float(time_step_fs)
    data["time_step_atomic"] = (
        float(time_step_fs) * FEMTOSECOND_TO_ATOMIC_TIME
    )
    return data
