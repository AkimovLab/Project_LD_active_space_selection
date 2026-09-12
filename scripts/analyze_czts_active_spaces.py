#!/usr/bin/env python3
"""Analyze fixed active spaces for the standalone 200 fs CZTS trajectory.

This script is intentionally separate from the manuscript workflow.  It reads
the tuple-encoded adiabatic Hamiltonians and real precomputed time overlaps,
applies the reusable numerical-support and LD convergence routines, writes
CZTS-only figures, and generates ``czts_analysis/CZTS_RESULTS.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from atomistic_data import prepare_text_atomistic_trajectory
from ld_active_space import (
    active_boundary_energy_gaps,
    cumulative_population_rmse,
    infer_constant_active_space,
    simulate_precomputed_overlap_ld,
    systematic_active_size_convergence,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
CZTS_DIRECTORY = (
    PROJECT_DIRECTORY / "Project_CZTS" / "CZTS_200fs_energy_timeoverlap"
)
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "czts_analysis"
RELATIVE_TOLERANCE = 1.0e-3
ABSOLUTE_TOLERANCE = 1.0e-12
HARTREE_TO_ELECTRONVOLT = 27.211386245988
INITIAL_STATES = (3, 14, 47, 79, 95, 114, 121)
REPRESENTATIVE_INITIAL_STATE = 79
REPRESENTATIVE_SIZES = (80, 81, 96, 97, 130)
LOW_STATE_INITIAL_STATE = 2
LOW_STATE_ACTIVE_SIZES = (4, 5, 6)


plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.6,
})


def support_margins(diagnostics):
    """Return the trajectory-minimum singular-value margin for each size."""

    return {
        size: float(np.min(item["sigma_min"] / item["cutoff"]))
        for size, item in diagnostics.items()
    }


def add_panel_labels(axes):
    """Add bold lower-case panel labels to a Matplotlib axes array."""

    for index, axis in enumerate(np.asarray(axes).ravel()):
        axis.text(
            0.018, 0.965, f"({chr(97 + index)})", transform=axis.transAxes,
            ha="left", va="top", fontsize=19, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            zorder=20,
        )


def save_figure(figure, filename):
    """Save and close one CZTS-only figure at publication-readable quality."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    path = OUTPUT_DIRECTORY / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def plot_support_selection(data, constant, convergence, gaps):
    """Plot full-space spectra, rank sensitivity, margins, and energy gaps."""

    spectra = constant["singular_values"]
    times = data["overlap_times_fs"]
    sizes = np.asarray(sorted(convergence["diagnostics"]))
    margins = support_margins(convergence["diagnostics"])
    margin_values = np.asarray([margins[size] for size in sizes])
    accepted = margin_values > 1.0

    figure, axes = plt.subplots(2, 2, figsize=(17, 13))
    axes[0, 0].semilogy(times, spectra[:, -1], label=r"$\sigma_{130}$")
    axes[0, 0].semilogy(times, spectra[:, -2], label=r"$\sigma_{129}$")
    axes[0, 0].semilogy(times, spectra[:, -5], label=r"$\sigma_{126}$")
    axes[0, 0].set(
        xlabel="Time-step interval start (fs)", ylabel="Singular value",
        title="Smallest full-space singular values",
    )
    axes[0, 0].grid(ls="--", alpha=0.25)
    axes[0, 0].legend(loc="lower right")

    for tolerance, color in zip((1.0e-4, 1.0e-3, 1.0e-2),
                                ("tab:green", "tab:blue", "tab:red")):
        cutoff = np.maximum(ABSOLUTE_TOLERANCE, tolerance * spectra[:, 0])
        ranks = np.sum(spectra > cutoff[:, None], axis=1)
        axes[0, 1].plot(
            times, ranks, color=color, label=rf"$\tau_{{rel}}={tolerance:g}$"
        )
    axes[0, 1].set(
        xlabel="Time-step interval start (fs)", ylabel="Reliable rank",
        title="Noise-threshold sensitivity", ylim=(0, data["overlaps"].shape[1] + 3),
    )
    axes[0, 1].grid(ls="--", alpha=0.25)
    axes[0, 1].legend(loc="lower left")

    axes[1, 0].plot(sizes, margin_values, color="0.45", zorder=1)
    axes[1, 0].scatter(
        sizes[accepted], margin_values[accepted], color="tab:blue", s=55,
        label="supported", zorder=3,
    )
    axes[1, 0].scatter(
        sizes[~accepted], margin_values[~accepted], color="tab:red", marker="x",
        linewidths=3.0, s=90, label="rejected", zorder=4,
    )
    axes[1, 0].axhline(1.0, color="black", ls="--", label="acceptance boundary")
    axes[1, 0].set(
        xlabel="Leading active-space size", ylabel="Support margin",
        title=rf"Fixed-block screen, $\tau_{{rel}}={RELATIVE_TOLERANCE:g}$",
        yscale="log",
    )
    axes[1, 0].grid(ls="--", alpha=0.25)
    axes[1, 0].legend(loc="upper right", ncol=2)

    gap_sizes = gaps["active_sizes"]
    gap_ev = gaps["minimum"] * HARTREE_TO_ELECTRONVOLT
    axes[1, 1].semilogy(gap_sizes, gap_ev, color="tab:purple", marker="o", ms=4)
    axes[1, 1].set(
        xlabel="Boundary after active-space size", ylabel="Minimum gap (eV)",
        title="Persistent energetic boundary guidance",
    )
    axes[1, 1].grid(ls="--", alpha=0.25)
    add_panel_labels(axes)
    save_figure(figure, "czts_active_space_selection.png")


def plot_population_convergence(convergence):
    """Plot population RMSE versus size for several fixed initial states."""

    diagnostics = convergence["diagnostics"]
    margins = support_margins(diagnostics)
    rejected = np.asarray([size for size, value in margins.items() if value <= 1.0])
    figure, axes = plt.subplots(2, 1, figsize=(15, 13), sharex=True)
    groups = (INITIAL_STATES[:4], INITIAL_STATES[4:])
    for axis, states in zip(axes, groups):
        for state in states:
            sizes = convergence["active_sizes"][state]
            rmse = convergence["rmse"][state]
            axis.semilogy(
                sizes, np.maximum(rmse, 1.0e-16), marker="o", ms=5,
                label=f"initial state {state}",
            )
        for size in rejected:
            axis.axvline(size, color="tab:red", lw=1.2, alpha=0.16)
        axis.grid(ls="--", alpha=0.25)
        axis.set_ylabel("Population RMSE vs 130 states")
        axis.legend(loc="upper right", ncol=2)
    axes[0].set_title("Lower and intermediate initial states")
    axes[1].set_title("Initial states near upper candidate boundaries")
    axes[1].set_xlabel("Leading active-space size")
    add_panel_labels(axes)
    save_figure(figure, "czts_population_convergence.png")


def representative_population_results(data):
    """Propagate selected spaces around two failed/recovered boundaries."""

    return {
        size: simulate_precomputed_overlap_ld(
            data["overlaps"], data["adiabatic_energies"],
            data["time_step_atomic"], size, REPRESENTATIVE_INITIAL_STATE,
            np.arange(REPRESENTATIVE_INITIAL_STATE + 1),
        )
        for size in REPRESENTATIVE_SIZES
    }


def low_state_population_results(data):
    """Propagate from state 2 in fixed active spaces of 4, 5, and 6 states.

    Parameters
    ----------
    data : dict
        Text atomistic trajectory returned by
        :func:`prepare_text_atomistic_trajectory`.

    Returns
    -------
    dict
        Results keyed by active-space size. Every calculation reports only
        populations of the common lowest states 0--3.
    """
    return {
        size: simulate_precomputed_overlap_ld(
            data["overlaps"], data["adiabatic_energies"],
            data["time_step_atomic"], size, LOW_STATE_INITIAL_STATE,
            np.arange(4),
        )
        for size in LOW_STATE_ACTIVE_SIZES
    }


def plot_low_state_active_space_populations(data, results):
    """Compare states 0--3 for the 4-, 5-, and 6-state active spaces.

    State identity is encoded by color, while active-space size is encoded by
    line style and marker. This makes differences caused by adding upper states
    visible without changing the visual identity of a reported state.
    """

    figure, axis = plt.subplots(figsize=(14, 8.5))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    styles = {
        4: dict(ls="-", marker="o", markevery=14),
        5: dict(ls="--", marker="s", markevery=14),
        6: dict(ls=":", marker="^", markevery=14),
    }
    for size in LOW_STATE_ACTIVE_SIZES:
        population = results[size]["reported_population"]
        for state, color in enumerate(colors):
            axis.plot(
                data["times_fs"], population[:, state], color=color,
                markerfacecolor="white", markeredgewidth=1.4,
                markersize=6.5, **styles[size],
            )
    axis.set(
        xlabel="Time (fs)", ylabel="Population",
        title="CZTS active-space comparison initialized in state 2",
        xlim=(data["times_fs"][0], data["times_fs"][-1]),
        ylim=(-0.02, 1.02),
    )
    axis.grid(ls="--", alpha=0.25)
    state_handles = [
        Line2D([0], [0], color=color, lw=3, label=f"State {state}")
        for state, color in enumerate(colors)
    ]
    size_handles = [
        Line2D(
            [0], [0], color="black", markerfacecolor="white",
            markeredgewidth=1.4, markersize=6.5,
            label=rf"$N_A={size}$", **styles[size],
        )
        for size in LOW_STATE_ACTIVE_SIZES
    ]
    state_legend = axis.legend(
        handles=state_handles, loc="upper right", ncol=2, title="Population"
    )
    axis.add_artist(state_legend)
    axis.legend(handles=size_handles, loc="center right", title="Active space")
    save_figure(figure, "czts_low_state_active_space_populations.png")


def plot_representative_populations(data, results, diagnostics):
    """Plot state populations and running error near problematic boundaries."""

    times = data["times_fs"]
    reference = results[130]
    figure, axes = plt.subplots(2, 2, figsize=(17, 13), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(REPRESENTATIVE_SIZES)))
    for color, size in zip(colors, REPRESENTATIVE_SIZES):
        accepted = diagnostics[size]["accepted"]
        label = rf"$N_A={size}$ ({'supported' if accepted else 'rejected'})"
        linestyle = "-" if accepted else "--"
        population = results[size]["reported_population"]
        axes[0, 0].plot(
            times, population[:, 79], color=color, ls=linestyle, label=label
        )
        axes[0, 1].plot(times, population[:, 78], color=color, ls=linestyle)
        axes[1, 0].plot(
            times, population[:, 78:80].sum(axis=1), color=color, ls=linestyle
        )
        axes[1, 1].semilogy(
            times, np.maximum(cumulative_population_rmse(results[size], reference),
                              1.0e-16),
            color=color, ls=linestyle,
        )
    axes[0, 0].set(title="Initially occupied state 79", ylabel="Population")
    axes[0, 1].set(title="Neighboring state 78", ylabel="Population")
    axes[1, 0].set(title="Combined states 78+79", ylabel="Population")
    axes[1, 1].set(
        title="Cumulative error relative to 130 states", ylabel="Population RMSE"
    )
    for axis in axes.ravel():
        axis.set_xlabel("Time (fs)")
        axis.grid(ls="--", alpha=0.25)
    axes[0, 0].legend(loc="upper right", ncol=2)
    add_panel_labels(axes)
    save_figure(figure, "czts_representative_populations.png")


def write_summary(
    data, constant, convergence, gaps, representative, low_state_results
):
    """Write a standalone Markdown interpretation of the CZTS calculations."""

    diagnostics = convergence["diagnostics"]
    margins = support_margins(diagnostics)
    accepted = [size for size in sorted(margins) if margins[size] > 1.0]
    rejected = [size for size in sorted(margins) if margins[size] <= 1.0]
    recovered = [size + 1 for size in rejected if margins.get(size + 1, 0.0) > 1.0]
    spectra = constant["singular_values"]
    threshold_ranks = {}
    for tolerance in (1.0e-4, 1.0e-3, 1.0e-2):
        cutoff = np.maximum(ABSOLUTE_TOLERANCE, tolerance * spectra[:, 0])
        rank = np.sum(spectra > cutoff[:, None], axis=1)
        threshold_ranks[tolerance] = (int(rank.min()), int(rank.max()))
    gap_order = np.argsort(gaps["minimum"])[::-1]
    leading_gaps = [
        (int(gaps["active_sizes"][index]),
         float(gaps["minimum"][index] * HARTREE_TO_ELECTRONVOLT))
        for index in gap_order[:8]
    ]
    reference = representative[130]
    representative_rmse = {
        size: float(cumulative_population_rmse(result, reference)[-1])
        for size, result in representative.items()
    }
    low_state_norm_errors = {
        size: np.max(np.abs(result["active_norm"] - 1.0))
        for size, result in low_state_results.items()
    }
    four_six_max_difference = np.max(np.abs(
        low_state_results[4]["reported_population"]
        - low_state_results[6]["reported_population"]
    ))
    lines = [
        "# CZTS Fixed-Active-Space LD Analysis",
        "",
        "This document is a standalone analysis and is not part of the paper manuscript.",
        "",
        "## Input and conventions",
        "",
        f"- Source: `{data['directory']}`",
        f"- Hamiltonian geometries: {len(data['indices'])} "
        f"(indices {data['indices'][0]}--{data['indices'][-1]})",
        f"- Propagated time-step intervals: {len(data['overlaps'])}",
        f"- Matrix dimension: {data['overlaps'].shape[1]}",
        f"- Saved-frame interval: {data['time_step_fs']:g} fs",
        f"- Unused overlap indices lacking a next Hamiltonian: "
        f"{data['unused_overlap_indices'].tolist()}",
        "- Hamiltonian diagonal values are treated as Hartree; overlaps are "
        "the supplied real adiabatic time-overlap matrices.",
        f"- Support cutoff: `max({ABSOLUTE_TOLERANCE:g}, "
        f"{RELATIVE_TOLERANCE:g} sigma_max)`.",
        "",
        "## Numerical support",
        "",
        f"At the working cutoff, the complete-space reliable rank is "
        f"{constant['step_reliable_rank'].min()}--"
        f"{constant['step_reliable_rank'].max()}, and the trajectory-wide "
        f"constant-rank bound is {constant['constant_rank']}. The smallest "
        f"complete-space singular value is {spectra[:, -1].min():.6g}, giving "
        f"a full-space support margin of "
        f"{margins[data['overlaps'].shape[1]]:.3f}.",
        "",
        f"Rejected leading dimensions are {rejected}. Their immediately larger "
        f"recovered dimensions are {recovered}. This nonmonotonic pattern means "
        "that a rejected boundary can cut through a mixed state group even when "
        "the larger completed block is well defined. It is therefore not valid "
        "to replace the full-space rank by the first `N` state labels.",
        "",
        "| Rejected size | Support margin | Recovered size | Support margin |",
        "|---:|---:|---:|---:|",
    ]
    for size in rejected:
        next_size = size + 1
        lines.append(
            f"| {size} | {margins[size]:.3f} | {next_size} | "
            f"{margins[next_size]:.3f} |"
        )
    lines.extend([
        "",
        "Threshold sensitivity of the complete-space rank:",
        "",
    ])
    for tolerance, rank_range in threshold_ranks.items():
        lines.append(
            f"- `tau_rel={tolerance:g}`: reliable-rank range "
            f"{rank_range[0]}--{rank_range[1]}"
        )
    lines.extend([
        "",
        "![CZTS numerical-support analysis](czts_active_space_selection.png)",
        "",
        "## Energetic guidance",
        "",
        "The strongest persistent boundary is after state 0 (`N_A=1`), followed "
        "by a substantial boundary after state 3 (`N_A=4`). Above this low-state "
        "group, trajectory-minimum gaps are much smaller, so energy gaps alone "
        "provide weak guidance for most of the 130-state manifold.",
        "",
        "| Boundary size | Minimum gap (eV) |",
        "|---:|---:|",
    ])
    for size, gap in leading_gaps:
        lines.append(f"| {size} | {gap:.6f} |")
    lines.extend([
        "",
        "## Population convergence",
        "",
        "Population convergence was evaluated retrospectively against the "
        "supported 130-state calculation for initial states 3, 14, 47, 79, 95, "
        "114, and 121. The curves are strongly dependent on how close the initial "
        "state lies to the upper boundary. Red vertical guides mark dimensions "
        "that fail the population-independent support test; their population "
        "errors are shown only as diagnostics and should not be used for production.",
        "",
        "![CZTS population convergence](czts_population_convergence.png)",
        "",
        "For initial state 79, the selected comparison brackets two problematic "
        "boundaries. Passing the support test "
        "does not guarantee agreement with the largest space: 81 and 97 states "
        "are numerically admissible, but their physical convergence must still be "
        "judged from the population comparison.",
        "",
        "| Active-space size | Support decision | Final cumulative RMSE |",
        "|---:|:---:|---:|",
    ])
    for size in REPRESENTATIVE_SIZES:
        lines.append(
            f"| {size} | "
            f"{'supported' if diagnostics[size]['accepted'] else 'rejected'} | "
            f"{representative_rmse[size]:.6e} |"
        )
    lines.extend([
        "",
        "For the low initial states, the error generally becomes small before the "
        "largest dimension, although the approach is not perfectly monotonic. "
        "For initial states close to the top of the computed manifold, especially "
        "states 114 and 121, no convincing pre-130 plateau appears. For those "
        "conditions the full supported space is the only defensible choice in this "
        "dataset; smaller accepted spaces are numerically valid but physically "
        "underconverged.",
        "",
        "![CZTS representative populations](czts_representative_populations.png)",
        "",
        "## Low-state active-space comparison",
        "",
        "Unitary polar-LD calculations were performed with 4-, 5-, and 6-state "
        "fixed active spaces, always with state 2 initially occupied. Only the "
        "common lowest four populations are compared. All three candidate "
        "blocks pass the trajectory-wide numerical-support test and conserve "
        "their full active-space norm to numerical precision.",
        "",
        "The color identifies the reported state in the figure, while line style "
        "and markers identify active-space size. Differences among curves of the "
        "same color directly show how states 4 and 5 alter the lower-state "
        "dynamics. The sum of the four displayed populations need not equal one "
        "in the 5- and 6-state runs, because a small population may reside in "
        "the additional active states. The maximum absolute population "
        f"difference between the 4- and 6-state curves is "
        f"{four_six_max_difference:.4f} over the trajectory.",
        "",
        "| Active size | State | Final population | Maximum population |",
        "|---:|---:|---:|---:|",
    ])
    for size in LOW_STATE_ACTIVE_SIZES:
        population = low_state_results[size]["reported_population"]
        for state in range(4):
            lines.append(
                f"| {size} | {state} | {population[-1, state]:.6f} | "
                f"{np.max(population[:, state]):.6f} |"
            )
    lines.extend([
        "",
        "| Active size | Support margin | Maximum norm error |",
        "|---:|---:|---:|",
    ])
    for size in LOW_STATE_ACTIVE_SIZES:
        lines.append(
            f"| {size} | {margins[size]:.3f} | "
            f"{low_state_norm_errors[size]:.2e} |"
        )
    lines.extend([
        "",
        "![CZTS low-state active-space population dynamics]"
        "(czts_low_state_active_space_populations.png)",
        "",
        "## Practical conclusion",
        "",
        f"The supplied CZTS trajectory supports the complete {data['overlaps'].shape[1]}-"
        "state LD space at `tau_rel=1e-3`; there is no numerical reason to clean "
        "it down to a smaller constant dimension at this cutoff. If a smaller "
        "space is required for cost, use persistent energetic/state-character "
        "boundaries as proposals, reject any candidate with support margin at or "
        "below one, and require population convergence across successive accepted "
        "sizes for every relevant initial state. In particular, avoid the six "
        f"unsupported cuts {rejected}; the recovered sizes {recovered} are better "
        "starting candidates, not automatic final choices. The conclusion is "
        "conditional on the chosen noise cutoff, which should ultimately be tied "
        "to the accuracy of the CZTS electronic-structure overlaps.",
        "",
    ])
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / "CZTS_RESULTS.md"
    path.write_text("\n".join(lines))
    print(f"Saved {path}")


def main():
    """Run the complete standalone CZTS selection and convergence workflow."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data = prepare_text_atomistic_trajectory(CZTS_DIRECTORY, time_step_fs=1.0)
    nstates = data["overlaps"].shape[1]
    active_sizes = list(range(1, nstates + 1))
    constant = infer_constant_active_space(
        data["overlaps"], RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE
    )
    convergence = systematic_active_size_convergence(
        data["overlaps"], data["adiabatic_energies"],
        data["time_step_atomic"], active_sizes, INITIAL_STATES,
        reference_size=nstates, relative_tolerance=RELATIVE_TOLERANCE,
    )
    gaps = active_boundary_energy_gaps(
        data["adiabatic_energies"], range(1, nstates)
    )
    representative = representative_population_results(data)
    low_state_results = low_state_population_results(data)
    plot_support_selection(data, constant, convergence, gaps)
    plot_population_convergence(convergence)
    plot_representative_populations(
        data, representative, convergence["diagnostics"]
    )
    plot_low_state_active_space_populations(data, low_state_results)
    write_summary(
        data, constant, convergence, gaps, representative, low_state_results
    )

    margins = support_margins(convergence["diagnostics"])
    rejected = [size for size in active_sizes if margins[size] <= 1.0]
    print("\nCZTS analysis summary")
    print(f"Geometries/intervals/dimension: {len(data['indices'])}/"
          f"{len(data['overlaps'])}/{nstates}")
    print(f"Complete-space constant reliable rank: {constant['constant_rank']}")
    print(f"Complete-space support margin: {margins[nstates]:.6f}")
    print(f"Rejected leading fixed sizes: {rejected}")


if __name__ == "__main__":
    main()
