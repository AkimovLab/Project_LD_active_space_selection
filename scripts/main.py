#!/usr/bin/env python3
"""Orchestrate the model and atomistic fixed-active-space LD calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atomistic_data import prepare_atomistic_trajectory
from ld_active_space import (
    clean_time_overlap,
    compare_fixed_active_spaces,
    convergence_diagnostics,
    diagnose_fixed_active_spaces,
    infer_constant_active_space,
    population_rmse,
    simulate_hamiltonian_active_space,
    simulate_precomputed_overlap_ld,
    systematic_active_size_convergence,
)
from models import build_model_trajectory, model_parameters
from paper_plots import (
    plot_atomistic_energy_blocks,
    plot_atomistic_selection,
    plot_atomistic_population_comparison,
    plot_dense_energy_blocks,
    plot_dense_overlap_diagnostics,
    plot_dense_population_convergence,
    plot_dense_timestep_diagnostics,
    plot_energy_gap_guidance,
    plot_fixed_space_comparison,
    plot_stress_diagnostics,
    plot_three_state_example,
    plot_unsupported_initial_state,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
FIGURES_DIRECTORY = SCRIPT_DIRECTORY / "figures"
CLEANING_RELATIVE_TOLERANCE = 1.0e-3

TIO2_DIRECTORY = PROJECT_DIRECTORY / "data" / "TiO2"
C20_DIRECTORY = PROJECT_DIRECTORY / "data" / "C20" / "res-tddft-energy"


def print_convergence_table(name, results, active_sizes):
    """Print nested-space population and overlap convergence diagnostics.

    Parameters
    ----------
    name : str
        Human-readable calculation label.
    results : dict
        Model propagation results keyed by active-space size.
    active_sizes : sequence of int
        Ordered sizes whose final entry is the internal full reference.
    """
    diagnostic = convergence_diagnostics(results, active_sizes)
    print(f"\n{name}")
    print("N_active  RMSE(full)  RMSE(next)  min sigma  min rank fraction")
    for index, size in enumerate(active_sizes):
        print(
            f"{size:8d}  {diagnostic['reference_rmse'][index]:10.3e}  "
            f"{diagnostic['adjacent_rmse'][index]:10.3e}  "
            f"{diagnostic['minimum_sigma'][index]:9.5f}  "
            f"{diagnostic['minimum_effective_rank_fraction'][index]:17.6f}"
        )


def print_time_overlap_summary(name, results, active_sizes):
    """Print the singular-value quantities used to screen candidate spaces.

    Parameters
    ----------
    name : str
        Human-readable calculation label.
    results : dict
        Model propagation results keyed by active size.
    active_sizes : sequence of int
        Candidate dimensions in display order.
    """
    print(f"\n{name}: time-overlap summary")
    print("N_active  min sigma  min effective-rank fraction")
    for size in active_sizes:
        result = results[size]
        fraction = result["effective_rank"] / size
        print(
            f"{size:8d}  {result['sigma_min'].min():9.5f}  "
            f"{fraction.min():27.6f}"
        )


def nonmonotonic_counts(convergence, initial_states):
    """Count upward population-RMSE steps as active dimension increases.

    Parameters
    ----------
    convergence : dict
        Output of ``systematic_active_size_convergence``.
    initial_states : iterable of int
        Initial-state series to summarize.

    Returns
    -------
    dict
        Initial states mapped to positive consecutive RMSE differences.
    """
    return {
        state: int(np.count_nonzero(np.diff(convergence["rmse"][state]) > 0.0))
        for state in initial_states
    }


def run_three_state_example():
    """Run and plot the full, nonunitary, and unitary two-state examples.

    Returns
    -------
    dict
        Model trajectory and all three propagation results.
    """
    trajectory = build_model_trajectory(model_parameters("three_state"))
    times = trajectory["times"]
    hamiltonian = trajectory["hamiltonian"]
    full = simulate_hamiltonian_active_space(
        hamiltonian, times, 3, 3, 0, np.arange(2)
    )
    truncated = simulate_hamiltonian_active_space(
        hamiltonian, times, 3, 2, 0, np.arange(2)
    )
    unitary_truncated = simulate_hamiltonian_active_space(
        hamiltonian, times, 3, 2, 0, np.arange(2), unitarize_overlap=True
    )
    plot_three_state_example(
        times, trajectory["energies"], full, truncated, unitary_truncated,
        CLEANING_RELATIVE_TOLERANCE, FIGURES_DIRECTORY,
    )
    print("\nThree-state truncation")
    print(f"Full-space minimum norm: {full['active_norm'].min():.6f}")
    print(f"Nonunitary two-state minimum norm: {truncated['active_norm'].min():.6f}")
    print(
        "Unitary-forced two-state maximum norm error: "
        f"{np.max(np.abs(unitary_truncated['active_norm'] - 1.0)):.3e}"
    )
    return {"trajectory": trajectory, "full": full, "truncated": truncated,
            "unitary_truncated": unitary_truncated}


def run_dense_model():
    """Run the ideal banded dense model and active-size convergence tests.

    Returns
    -------
    dict
        Dense trajectory, representative and systematic convergence series,
        and energy-gap/support inputs used by the paper figures.
    """
    params = model_parameters("dense_manifold", overlap_filter="none")
    trajectory = build_model_trajectory(params)
    times, hamiltonian = trajectory["times"], trajectory["hamiltonian"]
    nstates = params["nstates"]
    active_sizes = [6, 8, 10, 12, 14, 16, 20, 24, 30]
    results = {
        size: simulate_hamiltonian_active_space(
            hamiltonian, times, nstates, size, 5, np.arange(6),
            unitarize_overlap=True,
            propagation_time_step=trajectory["time_step_atomic"],
        ) for size in active_sizes
    }
    upper_sizes = [10, 12, 14, 16, 18, 20, 24, 30]
    upper_results = {
        size: simulate_hamiltonian_active_space(
            hamiltonian, times, nstates, size, 9, np.arange(10),
            unitarize_overlap=True,
            propagation_time_step=trajectory["time_step_atomic"],
        ) for size in upper_sizes
    }
    plot_dense_energy_blocks(
        times, trajectory["energies"], nstates,
        [(6, r"Closes band 1: $N_A=6$"), (8, r"Cuts band 2: $N_A=8$"),
         (10, r"Closes band 2: $N_A=10$"), (12, r"Cuts band 3: $N_A=12$")],
        FIGURES_DIRECTORY,
    )
    plot_dense_population_convergence(
        times,
        [
            {
                "results": results,
                "active_sizes": [6, 10, 14, 30],
                "states": (3, 4, 5),
                "initial_state": 5,
                "mixing_event_times": times[
                    np.abs(trajectory["exact_overlaps"][:, 4, 5]) > 0.5
                ],
            },
            {
                "results": upper_results,
                "active_sizes": [10, 12, 14, 30],
                "states": (6, 7, 9),
                "initial_state": 9,
            },
        ],
        FIGURES_DIRECTORY,
    )
    plot_dense_overlap_diagnostics(
        times, [(results, active_sizes, "initial state 5"),
                (upper_results, upper_sizes, "initial state 9")],
        CLEANING_RELATIVE_TOLERANCE, FIGURES_DIRECTORY,
    )
    timestep_cases = {
        params["time_step"]: {
            "trajectory": trajectory,
            "result": results[nstates],
        }
    }
    for step in (0.25, 0.125, 0.0625):
        refined_params = model_parameters(
            "dense_manifold", overlap_filter="none", time_step=step,
            time_stop=params["time_stop"],
        )
        refined_trajectory = build_model_trajectory(refined_params)
        refined_result = simulate_precomputed_overlap_ld(
            refined_trajectory["exact_overlaps"][1:],
            refined_trajectory["energies"],
            refined_trajectory["time_step_atomic"],
            nstates,
            5,
            np.arange(6),
        )
        timestep_cases[step] = {
            "trajectory": refined_trajectory,
            "result": refined_result,
        }
    plot_dense_timestep_diagnostics(timestep_cases, FIGURES_DIRECTORY)
    fine_step = min(timestep_cases)
    fine_population = timestep_cases[fine_step]["result"]["reported_population"]
    print("\nDense-manifold time-step convergence for states 4 and 5")
    print("dt (fs)  max abs error  RMS error  max combined-population error")
    for step in sorted(timestep_cases, reverse=True):
        if step == fine_step:
            continue
        population = timestep_cases[step]["result"]["reported_population"]
        stride = int(round(step / fine_step))
        reference = fine_population[::stride]
        difference = population[:, 4:6] - reference[:, 4:6]
        combined_difference = (
            population[:, 4:6].sum(axis=1)
            - reference[:, 4:6].sum(axis=1)
        )
        print(
            f"{step:7.4f}  {np.max(np.abs(difference)):13.6e}  "
            f"{np.sqrt(np.mean(difference**2)):9.3e}  "
            f"{np.max(np.abs(combined_difference)):29.6e}"
        )
    print_convergence_table("Dense manifold from state 5", results, active_sizes)
    print_convergence_table("Dense manifold from state 9", upper_results, upper_sizes)
    print_time_overlap_summary("Dense manifold from state 5", results, active_sizes)
    print_time_overlap_summary("Dense manifold from state 9", upper_results, upper_sizes)
    initial_states = [2, 5, 9, 13, 20]
    systematic = systematic_active_size_convergence(
        trajectory["exact_overlaps"][1:], trajectory["energies"],
        trajectory["time_step_atomic"], range(3, nstates + 1), initial_states,
    )
    gap_sizes = list(range(2, nstates))
    gap_diagnostics = diagnose_fixed_active_spaces(
        trajectory["exact_overlaps"][1:], gap_sizes,
        relative_tolerance=CLEANING_RELATIVE_TOLERANCE,
    )
    return {
        "params": params, "trajectory": trajectory,
        "active_sizes": active_sizes, "results": results,
        "upper_sizes": upper_sizes, "upper_results": upper_results,
        "systematic_initial_states": initial_states, "systematic": systematic,
        "gap_sizes": gap_sizes, "gap_diagnostics": gap_diagnostics,
        "timestep_cases": timestep_cases,
    }


def run_overlap_stress_test(dense_case, filter_name):
    """Apply one deterministic overlap filter to the dense-model trajectory.

    Parameters
    ----------
    dense_case : dict
        Ideal dense results returned by :func:`run_dense_model`.
    filter_name : {"hard_collapse", "ill_conditioned"}
        Filter selected through the model parameter dictionary.

    Returns
    -------
    dict
        Filtered data, support diagnostics, LD comparisons, and systematic
        active-size convergence.
    """
    if filter_name not in {"hard_collapse", "ill_conditioned"}:
        raise ValueError("Unknown overlap stress filter")
    params = model_parameters("dense_manifold", overlap_filter=filter_name)
    trajectory = build_model_trajectory(params)
    observed = trajectory["observed_overlaps"]
    cleaned = [clean_time_overlap(matrix, CLEANING_RELATIVE_TOLERANCE)
               for matrix in observed]
    singular_values = np.asarray([item["singular_values"] for item in cleaned])
    default_rank = np.asarray([np.linalg.matrix_rank(item) for item in observed])
    cleaned_rank = np.asarray([item["rank"] for item in cleaned])
    active_sizes = ([6, 10, 14, 18, 30] if filter_name == "hard_collapse"
                    else [6, 10, 11, 14, 30])
    diagnostics = diagnose_fixed_active_spaces(
        observed[1:], active_sizes,
        relative_tolerance=CLEANING_RELATIVE_TOLERANCE,
    )
    comparison = compare_fixed_active_spaces(
        trajectory["times"], trajectory["energies"],
        trajectory["exact_overlaps"], observed, active_sizes, 5, np.arange(6),
        time_step=trajectory["time_step_atomic"],
    )
    plot_stress_diagnostics(
        trajectory["times"], trajectory["energies"],
        trajectory["reliability_history"], singular_values, default_rank,
        cleaned_rank, diagnostics, active_sizes, filter_name, FIGURES_DIRECTORY,
    )
    boundary = None
    if filter_name == "ill_conditioned":
        boundary_sizes = [14, 30]
        boundary = compare_fixed_active_spaces(
            trajectory["times"], trajectory["energies"],
            trajectory["exact_overlaps"], observed, boundary_sizes, 13,
            np.arange(10, 14), time_step=trajectory["time_step_atomic"],
        )
        plot_unsupported_initial_state(
            trajectory["times"], boundary, boundary_sizes, FIGURES_DIRECTORY,
        )
    plot_fixed_space_comparison(
        trajectory["times"], comparison, diagnostics, filter_name,
        FIGURES_DIRECTORY,
    )
    systematic = systematic_active_size_convergence(
        observed[1:], trajectory["energies"], trajectory["time_step_atomic"],
        range(3, params["nstates"] + 1), dense_case["systematic_initial_states"],
        reference_overlaps=trajectory["exact_overlaps"][1:],
    )
    print(f"\n{filter_name.replace('_', ' ').title()}")
    print("Cleaned-rank range:", int(cleaned_rank.min()), "to", int(cleaned_rank.max()))
    print("Fixed-space acceptance:",
          {size: diagnostics[size]["accepted"] for size in active_sizes})
    print("Common-state RMSE versus ideal full reference:", {
        size: population_rmse(result, comparison["reference"])
        for size, result in comparison["results"].items()
    })
    if boundary is not None:
        print("Unsupported initial state 13:", {
            size: {"reference_RMSE": population_rmse(
                result, boundary["reference"])}
            for size, result in boundary["results"].items()
        })
        print("Default-rank range:", int(default_rank.min()), "to", int(default_rank.max()))
    return {
        "params": params, "trajectory": trajectory, "active_sizes": active_sizes,
        "diagnostics": diagnostics, "comparison": comparison,
        "boundary": boundary,
        "singular_values": singular_values, "default_rank": default_rank,
        "cleaned_rank": cleaned_rank, "systematic": systematic,
    }


def run_tio2_example():
    """Load, screen, propagate, and plot the atomistic TiO2 trajectory.

    Returns
    -------
    dict or None
        Complete analysis, or ``None`` if the source directory is absent.
    """
    if not TIO2_DIRECTORY.is_dir():
        print(f"TiO2 example skipped: data directory not found: {TIO2_DIRECTORY}")
        return None
    data = prepare_atomistic_trajectory(TIO2_DIRECTORY)
    active_sizes = [6, 8, 11]
    constant = infer_constant_active_space(
        data["overlaps"], CLEANING_RELATIVE_TOLERANCE
    )
    diagnostics = diagnose_fixed_active_spaces(
        data["overlaps"], active_sizes, CLEANING_RELATIVE_TOLERANCE
    )
    results = {
        size: simulate_precomputed_overlap_ld(
            data["overlaps"], data["adiabatic_energies"],
            data["time_step_atomic"], size, 5, np.arange(6),
        ) for size in active_sizes
    }
    initial_states = [1, 3, 5, 8]
    systematic = systematic_active_size_convergence(
        data["overlaps"], data["adiabatic_energies"], data["time_step_atomic"],
        range(2, data["overlaps"].shape[1] + 1), initial_states,
    )
    plot_atomistic_selection(
        data, constant, diagnostics, active_sizes, "TiO2", FIGURES_DIRECTORY
    )
    plot_atomistic_energy_blocks(
        data, active_sizes, constant["constant_rank"], FIGURES_DIRECTORY
    )
    print("\nTiO2 atomistic trajectory")
    print("Hamiltonian geometries:", len(data["indices"]),
          f"({data['indices'][0]} to {data['indices'][-1]})")
    print("Time overlaps used:", len(data["overlaps"]))
    print("Raw reliable-rank range:", int(constant["step_reliable_rank"].min()),
          "to", int(constant["step_reliable_rank"].max()))
    print("Selected constant rank:", constant["constant_rank"])
    print("Raw sigma_min range:",
          f"{constant['singular_values'][:, -1].min():.6f} to "
          f"{constant['singular_values'][:, -1].max():.6f}")
    for size in active_sizes[:-1]:
        print(f"LD common-state RMSE, N_A={size} versus N_A={active_sizes[-1]}:",
              f"{population_rmse(results[size], results[active_sizes[-1]]):.6e}")
    gap_sizes = list(range(2, data["overlaps"].shape[1]))
    gap_diagnostics = diagnose_fixed_active_spaces(
        data["overlaps"], gap_sizes, CLEANING_RELATIVE_TOLERANCE
    )
    return {
        "data": data, "active_sizes": active_sizes, "constant": constant,
        "diagnostics": diagnostics, "results": results,
        "systematic_initial_states": initial_states, "systematic": systematic,
        "gap_sizes": gap_sizes, "gap_diagnostics": gap_diagnostics,
    }


def run_c20_example():
    """Load, screen, propagate, and plot the atomistic C20 trajectory.

    Returns
    -------
    dict or None
        Complete analysis, or ``None`` if the source directory is absent.
    """
    if not C20_DIRECTORY.is_dir():
        print(f"C20 example skipped: data directory not found: {C20_DIRECTORY}")
        return None
    data = prepare_atomistic_trajectory(C20_DIRECTORY)
    nstates = data["overlaps"].shape[1]
    all_sizes = list(range(2, nstates + 1))
    constant = infer_constant_active_space(data["overlaps"], CLEANING_RELATIVE_TOLERANCE)
    diagnostics = diagnose_fixed_active_spaces(
        data["overlaps"], all_sizes, CLEANING_RELATIVE_TOLERANCE
    )
    accepted_sizes = [size for size in all_sizes if diagnostics[size]["accepted"]]
    reference_size = max(accepted_sizes)
    dynamics_sizes = [9, 10, 14, 41]
    results = {
        size: simulate_precomputed_overlap_ld(
            data["overlaps"], data["adiabatic_energies"],
            data["time_step_atomic"], size, 8, np.arange(9),
        ) for size in dynamics_sizes
    }
    initial_states = [1, 3, 8]
    systematic = systematic_active_size_convergence(
        data["overlaps"], data["adiabatic_energies"], data["time_step_atomic"],
        all_sizes, initial_states, reference_size=reference_size,
    )
    plot_atomistic_selection(
        data, constant, diagnostics, all_sizes, "C20", FIGURES_DIRECTORY
    )
    margins = {size: float(np.min(
        diagnostics[size]["sigma_min"] / diagnostics[size]["cutoff"]
    )) for size in all_sizes}
    print("\nC20 atomistic trajectory")
    print("Hamiltonian geometries:", len(data["indices"]),
          f"({data['indices'][0]} to {data['indices'][-1]})")
    print("Time overlaps used:", len(data["overlaps"]))
    print("Raw reliable-rank range:", int(constant["step_reliable_rank"].min()),
          "to", int(constant["step_reliable_rank"].max()))
    print("Intervals with deficient nominal full space:", int(np.count_nonzero(
        constant["step_reliable_rank"] < nstates)))
    print("Accepted leading fixed-block sizes:", accepted_sizes)
    print("Trajectory-minimum support margins:",
          {size: margins[size] for size in [5, 6, 9, 10, 14, 41]})
    print(f"C20 population RMSE relative to accepted N_A={reference_size}:", {
        size: population_rmse(result, results[reference_size])
        for size, result in results.items() if size != reference_size
    })
    print("C20 nonmonotonic RMSE increases:",
          nonmonotonic_counts(systematic, initial_states))
    return {
        "data": data, "all_sizes": all_sizes, "constant": constant,
        "diagnostics": diagnostics, "accepted_sizes": accepted_sizes,
        "reference_size": reference_size, "results": results,
        "systematic_initial_states": initial_states, "systematic": systematic,
        "gap_sizes": list(range(2, nstates)),
    }


def main():
    """Run every retained calculation and regenerate all curated figures."""
    run_three_state_example()
    dense = run_dense_model()
    hard = run_overlap_stress_test(dense, "hard_collapse")
    ill = run_overlap_stress_test(dense, "ill_conditioned")
    tio2 = run_tio2_example()
    c20 = run_c20_example()
    if tio2 is not None and c20 is not None:
        plot_atomistic_population_comparison(
            tio2["data"], tio2["results"], tio2["active_sizes"],
            c20["data"], c20["results"], [9, 10, 14, 41],
            c20["diagnostics"], FIGURES_DIRECTORY,
        )
        plot_energy_gap_guidance([
            (dense["trajectory"]["energies"], dense["gap_sizes"],
             dense["gap_diagnostics"], "Banded dense model",
             dense["params"]["band_boundaries"]),
            (tio2["data"]["adiabatic_energies"], tio2["gap_sizes"],
             tio2["gap_diagnostics"], "TiO$_2$", ()),
            (c20["data"]["adiabatic_energies"], c20["gap_sizes"],
             c20["diagnostics"], "C$_{20}$", ()),
        ], FIGURES_DIRECTORY)


if __name__ == "__main__":
    main()
