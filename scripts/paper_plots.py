"""Reusable plotting primitives and paper-oriented LD figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ld_active_space import active_boundary_energy_gaps


HARTREE_TO_ELECTRONVOLT = 27.211386245988


PAPER_FIGURES = {
    "few_state_truncation.png",
    "dense_model_energies.png",
    "dense_model_population_convergence.png",
    "dense_model_timestep_diagnostics.png",
    "dense_model_time_overlap_diagnostics.png",
    "tio2_constant_space_selection.png",
    "tio2_constant_space_energies.png",
    "atomistic_ld_populations.png",
    "severe_hard_rank_collapse_cleaning.png",
    "severe_hard_rank_collapse_ld_cleaning.png",
    "severe_ill_conditioned_overlap_cleaning.png",
    "severe_ill_conditioned_overlap_ld_cleaning.png",
    "severe_ill_conditioned_unsupported_initial_state.png",
    "c20_constant_space_selection.png",
    "energy_gap_candidate_guidance.png",
}


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.8,
})


def add_panel_labels(axes, fontsize=20):
    """Add bold journal-style labels to one axis or an array of axes.

    Parameters
    ----------
    axes : matplotlib axis or array_like of axes
        Panels labeled in flattened row-major order.
    fontsize : float, optional
        Minimum label size in points.
    """
    for index, axis in enumerate(np.atleast_1d(axes).ravel()):
        axis.text(
            0.015, 0.965, f"({chr(97 + index)})",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                  "pad": 1.5},
            zorder=20,
        )


def apply_paper_figure_style(figure):
    """Enforce consistent large typography, markers, and strokes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Completed figure whose artists are modified in place before saving.
    """
    for axis in figure.axes:
        axis.title.set_fontsize(max(axis.title.get_fontsize(), 22))
        axis.xaxis.label.set_fontsize(max(axis.xaxis.label.get_fontsize(), 20))
        axis.yaxis.label.set_fontsize(max(axis.yaxis.label.get_fontsize(), 20))
        axis.tick_params(axis="both", labelsize=17, width=1.6, length=6.0)
        for line in axis.lines:
            line.set_linewidth(max(line.get_linewidth(), 3.0))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(line.get_markersize(), 8.0))
        for collection in axis.collections:
            if hasattr(collection, "get_sizes"):
                sizes = collection.get_sizes()
                if len(sizes):
                    collection.set_sizes(np.maximum(sizes, 58.0))
        for annotation in axis.texts:
            annotation.set_fontsize(max(annotation.get_fontsize(), 18))
        legend = axis.get_legend()
        if legend is not None:
            for text_item in legend.get_texts():
                text_item.set_fontsize(max(text_item.get_fontsize(), 17))
            for handle in legend.get_lines():
                handle.set_linewidth(max(handle.get_linewidth(), 3.5))
                if handle.get_marker() not in (None, "None", "", " "):
                    handle.set_markersize(max(handle.get_markersize(), 8.0))
    if figure._suptitle is not None:
        figure._suptitle.set_fontsize(max(figure._suptitle.get_fontsize(), 23))


def save_paper_figure(figure, filename, figures_directory):
    """Style and save one curated paper figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save and close.
    filename : str
        Curated filename listed in :data:`PAPER_FIGURES`.
    figures_directory : str or pathlib.Path
        Output folder, created if necessary.

    Returns
    -------
    pathlib.Path or None
        Saved path, or ``None`` when a non-curated exploratory filename is
        intentionally suppressed.
    """
    if filename not in PAPER_FIGURES:
        plt.close(figure)
        return None
    directory = Path(figures_directory)
    directory.mkdir(parents=True, exist_ok=True)
    apply_paper_figure_style(figure)
    output = directory / filename
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")
    return output


def plot_population_family(
    axis, times, results, active_sizes, state, styles=None, reference_size=None
):
    """Plot one state's population for a family of active-space calculations.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Destination panel.
    times : array_like
        Geometry times.
    results : dict
        Propagation results keyed by active-space size.
    active_sizes : sequence of int
        Curves to display in order.
    state : int
        Column index in each result's ``reported_population`` array.
    styles : dict, optional
        Mapping ``size -> (color, linestyle)``.  A viridis palette is generated
        when omitted.
    reference_size : int, optional
        Size labeled ``full space`` and emphasized in black.

    Returns
    -------
    matplotlib.axes.Axes
        The modified destination axis.
    """
    if styles is None:
        colors = plt.cm.viridis(np.linspace(0.03, 0.92, len(active_sizes)))
        styles = {size: (color, "-") for size, color in zip(active_sizes, colors)}
    for size in active_sizes:
        color, linestyle = styles[size]
        is_reference = size == reference_size
        axis.plot(
            times, results[size]["reported_population"][:, state],
            color="black" if is_reference else color,
            ls="--" if is_reference else linestyle,
            lw=3.0 if is_reference else 2.2,
            label="full space" if is_reference else rf"$N_A={size}$",
        )
    return axis


def plot_support_margin(axis, times, diagnostics, active_sizes):
    """Plot numerical-support margins for several candidate blocks.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Destination panel.
    times : array_like
        Times assigned to the time-step intervals.
    diagnostics : dict
        Output of ``diagnose_fixed_active_spaces``.
    active_sizes : sequence of int
        Candidate dimensions to display.

    Returns
    -------
    matplotlib.axes.Axes
        Modified axis with the acceptance threshold at one.
    """
    for size in active_sizes:
        item = diagnostics[size]
        axis.semilogy(
            times, item["sigma_min"] / item["cutoff"],
            lw=2.5, label=rf"$N_A={size}$",
        )
    axis.axhline(1.0, color="black", ls=":", lw=2.2)
    axis.set(
        ylabel=r"$\sigma_{\min}/\tau_{\mathrm{clean}}$",
        title="Numerical-support margin of candidate fixed spaces",
    )
    return axis


def plot_three_state_example(
    times, energies, full, truncated, unitary_truncated, cutoff, output_directory
):
    """Create the paper figure for the three-state truncation failure.

    Parameters
    ----------
    times, energies : array_like
        Model times and three adiabatic energy branches.
    full, truncated, unitary_truncated : dict
        Full three-state, direct two-state contraction, and polar-unitarized
        two-state propagation results.
    cutoff : float
        Relative effective-rank threshold shown in the diagnostic panels.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(2, 2, figsize=(18, 13.5), sharex=True)
    cutoff_exponent = int(np.rint(np.log10(cutoff)))
    cutoff_label = rf"$10^{{{cutoff_exponent}}}\sigma_{{max}}$"
    colors = ("tab:blue", "tab:orange", "tab:green")
    for state, color in enumerate(colors):
        axes[0, 0].plot(
            times, HARTREE_TO_ELECTRONVOLT * energies[:, state],
            color=color, lw=4.0,
                        label=f"state {state}")
    axes[0, 0].set(ylabel="Adiabatic energy (eV)",
                   title="Three-state crossing model")
    axes[0, 0].legend(loc="best")
    for state in (0, 1):
        axes[0, 1].plot(times, full["active_population"][:, state],
                        color=colors[state], label=f"state {state}, full")
        axes[0, 1].plot(
            times, truncated["active_population"][:, state],
            color=colors[state], ls="none", marker="o", markevery=4, ms=7,
            label=f"state {state}, nonunitary",
        )
        axes[0, 1].plot(
            times, unitary_truncated["active_population"][:, state],
            color=colors[state], ls="--",
            label=f"state {state}, unitary-forced",
        )
    axes[0, 1].plot(times, full["active_population"][:, 2],
                    color=colors[2], label="state 2, full")
    axes[0, 1].set(ylabel="Population", title="Full and two-state-truncated LD",
                   ylim=(-0.03, 1.03))
    axes[0, 1].legend(loc="center", bbox_to_anchor=(0.55, 0.48), ncol=2)
    deficient = truncated["effective_rank"] < 2
    axes[1, 0].plot(times, full["sigma_min"], color="black",
                    label="full three-state space")
    axes[1, 0].axhline(cutoff, color="0.45", ls="--",
                       label="numerical cutoff " + cutoff_label)
    axes[1, 0].scatter(times[::4], truncated["sigma_min"][::4],
                       color="tab:purple", s=65, label="two-state truncation")
    axes[1, 0].scatter(
        times[deficient], truncated["sigma_min"][deficient],
        facecolor="tab:purple", edgecolor="tab:red", linewidth=2.8, s=175,
        label="deficient truncated interval",
    )
    axes[1, 0].set(xlabel="Time (a.u.)", ylabel=r"$\sigma_{\min}(S_A)$",
                   title="Worst retained singular direction", ylim=(-0.03, 1.04))
    axes[1, 0].legend(loc="center", bbox_to_anchor=(0.53, 0.54), ncol=2)
    axes[1, 1].plot(times, full["effective_rank"] / 3, color="black",
                    label="full three-state space")
    rank_fraction = truncated["effective_rank"] / 2
    axes[1, 1].scatter(times[::4], rank_fraction[::4], color="tab:purple",
                       s=65, label="two-state truncation")
    axes[1, 1].scatter(
        times[rank_fraction < 1], rank_fraction[rank_fraction < 1],
        facecolor="tab:purple", edgecolor="tab:red", linewidth=2.8, s=175,
        label="deficient truncated interval",
    )
    axes[1, 1].set(xlabel="Time (a.u.)", ylabel="Effective-rank fraction",
                   title="Rank fraction above " + cutoff_label,
                   ylim=(-0.03, 1.05))
    axes[1, 1].legend(loc="lower center", bbox_to_anchor=(0.53, 0.08), ncol=2)
    for axis in axes.flat:
        axis.grid(ls="--", lw=1.3, alpha=0.30)
    add_panel_labels(axes, fontsize=22)
    figure.tight_layout(pad=1.5)
    save_paper_figure(figure, "few_state_truncation.png", output_directory)


def plot_dense_energy_blocks(
    times, energies, nstates, panels, output_directory
):
    """Plot included and excluded energy branches for selected dense blocks.

    Parameters
    ----------
    times, energies : array_like
        Dense-model time grid and energy branches.
    nstates : int
        Full state count.
    panels : sequence of (int, str)
        Active size and descriptive title for each of four panels.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    for axis, (size, title) in zip(axes.flat, panels):
        for state in range(nstates):
            included = state < size
            axis.plot(
                times, HARTREE_TO_ELECTRONVOLT * energies[:, state],
                color="tab:blue" if included else "0.70",
                lw=2.7 if included else 1.4,
                alpha=0.95 if included else 0.65,
                label=("active" if state == 0 else
                       "excluded" if state == size else None),
            )
        axis.set(ylabel="Adiabatic energy (eV)", title=title)
        axis.grid(ls="--", alpha=0.25)
        axis.legend(loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("Time (fs)")
    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(figure, "dense_model_energies.png", output_directory)


def plot_dense_population_convergence(
    times, cases, output_directory,
):
    """Plot the state-5 and state-9 dense-model examples in a 2-by-3 grid.

    Parameters
    ----------
    times : array_like
        Model times.
    cases : sequence of dict
        Two row specifications. Each dictionary contains ``results`` keyed by
        active-space size, the displayed ``active_sizes``, three ``states``, an
        ``initial_state``, and optional ``mixing_event_times``.
    output_directory : path_like
        Figure destination.

    Notes
    -----
    The first row contains the three individual populations for the state-5
    initialization; the earlier combined ``P4 + P5`` panel is intentionally
    omitted. The second row contains the three state-9 panels.
    """
    if len(cases) != 2 or any(len(case["states"]) != 3 for case in cases):
        raise ValueError("Exactly two cases with three states each are required")
    figure, axes = plt.subplots(
        2, 3, figsize=(24.0, 15.5), sharex=True, sharey=True,
    )
    for row, case in enumerate(cases):
        active_sizes = case["active_sizes"]
        for column, state in enumerate(case["states"]):
            axis = axes[row, column]
            plot_population_family(
                axis, times, case["results"], active_sizes, state,
                reference_size=active_sizes[-1],
            )
            axis.set_title(
                f"Initial state {case['initial_state']}: state {state}",
                fontsize=30,
            )
            axis.tick_params(axis="both", labelsize=29, width=2.2, length=8.0)
            for line in axis.lines:
                line.set_linewidth(5.0 if line.get_label() == "full space" else 4.3)
            axis.grid(ls="--", alpha=0.25)
            for event_time in case.get("mixing_event_times", ()):
                axis.axvline(
                    event_time, color="0.48", ls=":", lw=3.2, alpha=0.7
                )
        axes[row, 0].set_ylabel("Population", fontsize=29)
        axes[row, 0].legend(loc="best", ncol=2, fontsize=23)
    for axis in axes[-1]:
        axis.set_xlabel("Time (fs)", fontsize=29)
    add_panel_labels(axes, fontsize=29)
    figure.tight_layout(h_pad=2.0, w_pad=1.4)
    save_paper_figure(
        figure, "dense_model_population_convergence.png", output_directory
    )


def plot_dense_timestep_diagnostics(cases, output_directory):
    """Diagnose the rapid state-4/5 population changes and LD time step.

    Parameters
    ----------
    cases : dict
        Mapping from time-step interval in femtoseconds to dictionaries with
        ``trajectory`` and ``result`` entries.  The result must report states
        0--5 for full 30-state unitary LD initialized in state 5.
    output_directory : path_like
        Destination for the curated paper figure.

    Notes
    -----
    The finest supplied interval is used as the numerical reference.  The
    overlap panel deliberately reports the 0.5 fs matrix because an overlap
    element measures rotation accumulated over its stated interval and cannot
    be compared directly across unequal intervals.
    """
    steps = sorted(cases, reverse=True)
    coarse_step = max(cases)
    fine_step = min(cases)
    coarse = cases[coarse_step]
    fine = cases[fine_step]
    fine_times = np.asarray(fine["trajectory"]["times"])
    fine_population = np.asarray(fine["result"]["reported_population"])

    figure, axes = plt.subplots(2, 2, figsize=(17.0, 13.0))

    energy = np.asarray(fine["trajectory"]["energies"])
    gap_mev = (
        energy[:, 5] - energy[:, 4]
    ) * HARTREE_TO_ELECTRONVOLT * 1000.0
    axes[0, 0].plot(fine_times, gap_mev, color="tab:purple")
    axes[0, 0].set(
        xlabel="Time (fs)", ylabel=r"$E_5-E_4$ (meV)",
        title="Near-degeneracy of states 4 and 5",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(ls="--", alpha=0.25)

    coarse_times = np.asarray(coarse["trajectory"]["times"])
    overlap = np.abs(np.asarray(coarse["trajectory"]["exact_overlaps"])[1:])
    interval_times = 0.5 * (coarse_times[:-1] + coarse_times[1:])
    axes[0, 1].plot(interval_times, overlap[:, 4, 4], label=r"$|S_{44}|$")
    axes[0, 1].plot(interval_times, overlap[:, 4, 5], label=r"$|S_{45}|$")
    axes[0, 1].plot(interval_times, overlap[:, 5, 5], label=r"$|S_{55}|$")
    axes[0, 1].set(
        xlabel="Time (fs)", ylabel="Overlap magnitude",
        title=rf"Adiabatic rotation over a {coarse_step:g} fs interval",
    )
    axes[0, 1].grid(ls="--", alpha=0.25)
    axes[0, 1].legend(loc="lower right", ncol=3)

    coarse_population = np.asarray(coarse["result"]["reported_population"])
    axes[1, 0].plot(
        coarse_times, coarse_population[:, 4], color="tab:blue",
        label=rf"State 4, $\Delta t={coarse_step:g}$ fs",
    )
    axes[1, 0].plot(
        coarse_times, coarse_population[:, 5], color="tab:red",
        label=rf"State 5, $\Delta t={coarse_step:g}$ fs",
    )
    marker_stride = max(1, int(round(1.0 / fine_step)))
    axes[1, 0].plot(
        fine_times[::marker_stride], fine_population[::marker_stride, 4],
        ls="none", marker="o", ms=6.5, mfc="none", mec="tab:blue",
        label=rf"State 4, $\Delta t={fine_step:g}$ fs",
    )
    axes[1, 0].plot(
        fine_times[::marker_stride], fine_population[::marker_stride, 5],
        ls="none", marker="s", ms=6.5, mfc="none", mec="tab:red",
        label=rf"State 5, $\Delta t={fine_step:g}$ fs",
    )
    axes[1, 0].set(
        xlabel="Time (fs)", ylabel="Population",
        title="Population exchange and fine-step reference",
    )
    axes[1, 0].grid(ls="--", alpha=0.25)
    axes[1, 0].legend(loc="upper right", ncol=2)

    for step in steps:
        if step == fine_step:
            continue
        case_times = np.asarray(cases[step]["trajectory"]["times"])
        population = np.asarray(cases[step]["result"]["reported_population"])
        ratio = int(round(step / fine_step))
        reference = fine_population[::ratio]
        if len(reference) != len(population):
            raise ValueError("Time-step grids must have matching endpoints")
        error = np.max(np.abs(population[:, 4:6] - reference[:, 4:6]), axis=1)
        axes[1, 1].semilogy(
            case_times, np.maximum(error, 1.0e-16),
            label=rf"$\Delta t={step:g}$ fs",
        )
    axes[1, 1].set(
        xlabel="Time (fs)", ylabel=r"max $|\Delta P_{4,5}|$",
        title=rf"Error relative to $\Delta t={fine_step:g}$ fs",
    )
    axes[1, 1].grid(ls="--", alpha=0.25)
    axes[1, 1].legend(loc="lower right")

    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(
        figure, "dense_model_timestep_diagnostics.png", output_directory
    )


def plot_dense_overlap_diagnostics(
    times, cases, relative_tolerance, output_directory
):
    """Compare the overlap-support diagnostics for two dense-model series.

    Parameters
    ----------
    times : array_like
        Model geometry times.
    cases : sequence of (dict, sequence, str)
        Results, active sizes, and column title for two initial-state series.
    relative_tolerance : float
        Singular-value cutoff drawn in the first row.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(2, 2, figsize=(19, 13), sharex="col", sharey="row")
    for column, (results, active_sizes, title) in enumerate(cases):
        colors = plt.cm.turbo(np.linspace(0.02, 0.92, len(active_sizes)))
        for color, size in zip(colors, active_sizes):
            axes[0, column].plot(times, results[size]["sigma_min"], color=color,
                                 label=rf"$N_A={size}$")
            axes[1, column].plot(times, results[size]["effective_rank"] / size,
                                 color=color)
        axes[0, column].axhline(relative_tolerance, color="black", ls=":")
        axes[0, column].set(yscale="log", ylim=(5.0e-4, 1.2),
                            title=title + ": minimum singular value")
        axes[1, column].axhline(1.0, color="black", ls=":")
        axes[1, column].set(title=title + ": effective-rank fraction")
        axes[1, column].set_xlabel("Time (fs)")
        axes[0, column].legend(ncol=2)
    axes[0, 0].set_ylabel(r"$\sigma_{\min}(S_A)$")
    axes[1, 0].set_ylabel(r"$r_{\mathrm{eff}}/N_A$")
    for axis in axes.flat:
        axis.grid(ls="--", alpha=0.25)
    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(
        figure, "dense_model_time_overlap_diagnostics.png", output_directory
    )


def plot_stress_diagnostics(
    times, reference_energies, reliability, singular_values, default_rank,
    cleaned_rank, diagnostics, active_sizes, filter_name, output_directory,
):
    """Plot overlap-support diagnostics for one synthetic stress filter.

    Parameters
    ----------
    times, reference_energies : array_like
        Geometry times and unmodified Hamiltonian energies.  The latter are
        reference labels only because stress overlaps are altered independently.
    reliability, singular_values : array_like
        Synthetic factors and observed-overlap singular spectra.
    default_rank, cleaned_rank : array_like
        Formal and noise-informed ranks at every geometry-indexed matrix.
    diagnostics : dict
        Candidate-block support diagnostics computed on propagation intervals.
    active_sizes : sequence of int
        Candidate dimensions.
    filter_name : {"hard_collapse", "ill_conditioned"}
        Selects presentation details and output filename.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(4, 1, figsize=(13, 18), sharex=True)
    nstates = reference_energies.shape[1]
    if filter_name == "hard_collapse":
        for state in range(nstates):
            axes[0].plot(
                times, HARTREE_TO_ELECTRONVOLT * reference_energies[:, state],
                color="0.72", lw=1.5,
            )
            attenuated = reliability[:, state] < 1.0e-8
            axes[0].scatter(
                times[attenuated],
                HARTREE_TO_ELECTRONVOLT * reference_energies[attenuated, state],
                            color="tab:red", s=12)
        axes[0].set(title="Reference energies and attenuated indices (not a gap test)")
        vmin, filename = -16, "severe_hard_rank_collapse_cleaning.png"
    else:
        log_reliability = np.log10(np.maximum(reliability, 1.0e-15))
        for state in range(nstates):
            points = axes[0].scatter(
                times, HARTREE_TO_ELECTRONVOLT * reference_energies[:, state],
                c=log_reliability[:, state], cmap="magma", vmin=-15, vmax=0,
                s=12, linewidths=0,
            )
        figure.colorbar(points, ax=axes[0], label=r"$\log_{10}$ reliability factor")
        axes[0].set(title="Reference energies and index attenuation (not a gap test)")
        vmin, filename = -15, "severe_ill_conditioned_overlap_cleaning.png"
    axes[0].set_ylabel("Reference energy (eV)")
    image = axes[1].imshow(
        np.log10(np.maximum(singular_values.T, 1.0e-16)), origin="lower",
        aspect="auto", extent=(times[0], times[-1], 0, nstates - 1),
        cmap="magma", vmin=vmin, vmax=0,
    )
    axes[1].set(ylabel="Singular-value index",
                title="Singular spectrum of the observed overlap")
    figure.colorbar(image, ax=axes[1], label=r"$\log_{10}\sigma_k$")
    axes[2].plot(times, default_rank, color="black", label="NumPy default rank")
    axes[2].plot(times, cleaned_rank, color="tab:blue", ls="--",
                 label=r"cleaned rank, $\tau_{rel}=10^{-3}$")
    axes[2].set(ylabel="Retained dimension",
                title="Rank selected by the implemented singular-value cutoff")
    axes[2].legend()
    plot_support_margin(axes[3], times[1:], diagnostics, active_sizes)
    axes[3].set_xlabel("Time (fs)")
    axes[3].legend(ncol=3)
    for axis in axes:
        axis.grid(ls="--", alpha=0.3)
    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(figure, filename, output_directory)


def plot_fixed_space_comparison(
    times, comparison, diagnostics, filter_name, output_directory,
):
    """Plot representative populations for fixed-space stress tests.

    Parameters
    ----------
    times : array_like
        Model geometry times.
    comparison : dict
        Candidate and reference results from ``compare_fixed_active_spaces``.
    diagnostics : dict
        Candidate support decisions.
    filter_name : {"hard_collapse", "ill_conditioned"}
        Selects title and curated filename.
    output_directory : path_like
        Figure destination.
    """
    representative = (3, 4, 5)
    sizes = sorted(comparison["results"])
    colors = plt.cm.viridis(np.linspace(0.05, 0.90, len(sizes)))
    figure, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True)
    for axis, state in zip(axes, representative):
        for color, size in zip(colors, sizes):
            accepted = diagnostics[size]["accepted"]
            axis.plot(
                times, comparison["results"][size]["reported_population"][:, state],
                color=color, ls="-" if accepted else "--",
                label=rf"$N_A={size}$ ({'accepted' if accepted else 'rejected'})",
            )
        axis.plot(
            times, comparison["reference"]["reported_population"][:, state],
            color="black", ls="none", marker="o", markevery=2, ms=6.5,
            zorder=15, label="ideal full-space reference (dots)",
        )
        axis.set(ylabel="Population", title=f"Adiabatic state {state}")
        axis.grid(ls="--", alpha=0.25)
    axes[-1].set_xlabel("Time (fs)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]],
                   ncol=2, loc="upper right")
    title = ("Hard rank collapse" if filter_name == "hard_collapse"
             else "Ill-conditioned overlap")
    figure.suptitle(title + ": fixed-space unitary local diabatization")
    add_panel_labels(axes)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    filename = (
        "severe_hard_rank_collapse_ld_cleaning.png"
        if filter_name == "hard_collapse"
        else "severe_ill_conditioned_overlap_ld_cleaning.png"
    )
    save_paper_figure(figure, filename, output_directory)


def plot_unsupported_initial_state(
    times, comparison, active_sizes, output_directory
):
    """Plot ill-conditioned dynamics initialized in an unsupported direction.

    Parameters
    ----------
    times : array_like
        Model times.
    comparison : dict
        Ideal and observed-overlap fixed-space comparisons.
    active_sizes : sequence of int
        Rejected sizes directly probed by the boundary initial state.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    colors = {14: "tab:orange", 30: "tab:red"}
    for axis, state, column in ((axes[0], 13, 3), (axes[1], 10, 0)):
        for size in active_sizes:
            axis.plot(times, comparison["results"][size]["reported_population"][:, column],
                      color=colors[size], ls="--", label=rf"$N_A={size}$ (rejected)")
        axis.plot(
            times, comparison["reference"]["reported_population"][:, column],
            color="black", ls="none", marker="o", markevery=2, ms=6.5,
            zorder=15, label="ideal full-space reference (dots)",
        )
        axis.set(title=f"Adiabatic state {state}", ylabel="Population")
        axis.grid(ls="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]],
                   loc="best")
    for axis in axes:
        axis.set_xlabel("Time (fs)")
    figure.suptitle("Ill-conditioned overlap initialized in unsupported state 13")
    add_panel_labels(axes)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_paper_figure(
        figure, "severe_ill_conditioned_unsupported_initial_state.png",
        output_directory,
    )


def plot_atomistic_selection(
    data, constant, diagnostics, active_sizes, system_name, output_directory
):
    """Plot trajectory-wide support diagnostics for an atomistic dataset.

    Parameters
    ----------
    data : dict
        Prepared atomistic trajectory.
    constant : dict
        Full-matrix trajectory-wide rank analysis.
    diagnostics : dict
        Candidate-block diagnostics.
    active_sizes : sequence of int
        Candidate dimensions.  For C20 this should contain every tested size.
    system_name : {"TiO2", "C20"}
        Selects the system-specific paper layout.
    output_directory : path_like
        Figure destination.
    """
    if system_name == "TiO2":
        figure, axes = plt.subplots(3, 1, figsize=(13, 18), sharex=True)
        image = axes[0].imshow(
            constant["singular_values"].T, origin="lower", aspect="auto",
            extent=(data["overlap_times_fs"][0], data["overlap_times_fs"][-1],
                    0, data["overlaps"].shape[1] - 1), cmap="viridis",
            vmin=min(0.65, constant["singular_values"].min()),
            vmax=max(1.20, constant["singular_values"].max()),
        )
        axes[0].set(ylabel="Singular-value index",
                    title="Raw TiO$_2$ time-overlap singular spectrum")
        figure.colorbar(image, ax=axes[0], label=r"$\sigma_k$")
        margin = constant["singular_values"][:, -1] / constant["step_cutoff"]

        """
        axes[1].semilogy(data["overlap_times_fs"], margin, color="tab:purple")
        axes[1].axhline(1.0, color="black", ls=":")
        axes[1].set(ylabel=r"$\sigma_{\min}/\tau_{\mathrm{clean}}$",
                    title="Margin above the retain/discard cutoff")
        """
        axes[1].plot(data["overlap_times_fs"], constant["step_reliable_rank"],
                     color="black", label="per-interval reliable rank")
        axes[1].axhline(constant["constant_rank"], color="tab:blue", ls="--",
                        label="trajectory-wide constant rank")
        axes[1].set(ylabel="Retained dimension",
                    title="Maximal rank valid at every time-step interval")
        axes[1].legend(loc="lower right")
        plot_support_margin(axes[2], data["overlap_times_fs"], diagnostics, active_sizes)
        axes[2].set_xlabel("Time (fs)")
        axes[2].legend()
        filename = "tio2_constant_space_selection.png"
    else:
        nstates = data["overlaps"].shape[1]
        reference_size = max(size for size in active_sizes if diagnostics[size]["accepted"])
        figure, axes = plt.subplots(2, 2, figsize=(19, 14))
        for state in range(nstates):
            axes[0, 0].plot(
                data["times_fs"],
                HARTREE_TO_ELECTRONVOLT * data["adiabatic_energies"][:, state],
                color="tab:blue" if state < reference_size else "0.72",
                lw=2.4 if state < reference_size else 1.2,
                label=(rf"accepted $N_A={reference_size}$ block" if state == 0
                       else "states outside that block" if state == reference_size
                       else None),
            )
        axes[0, 0].set(xlabel="Time (fs)", ylabel="Adiabatic energy (eV)",
                       title="C$_{20}$ adiabatic CI-state energies")
        axes[0, 0].legend()
        image = axes[0, 1].imshow(
            constant["singular_values"].T, origin="lower", aspect="auto",
            extent=(data["overlap_times_fs"][0], data["overlap_times_fs"][-1],
                    0, nstates - 1), cmap="viridis",
            norm=LogNorm(vmin=constant["singular_values"].min(),
                         vmax=constant["singular_values"].max()),
        )
        axes[0, 1].set(xlabel="Time (fs)", ylabel="Singular-value index",
                       title="Full raw time-overlap singular spectrum")
        figure.colorbar(image, ax=axes[0, 1], label=r"$\sigma_k$")
        full_margin = constant["singular_values"][:, -1] / constant["step_cutoff"]
        axes[1, 0].semilogy(data["overlap_times_fs"], full_margin,
                           color="tab:purple")
        axes[1, 0].axhline(1.0, color="black", ls=":")
        deficient_count = np.count_nonzero(
            constant["step_reliable_rank"] < nstates
        )
        axes[1, 0].set(
            xlabel="Time (fs)", ylabel=r"$\sigma_{\min}/\tau_{\mathrm{clean}}$",
            title=f"Full-space support margin (deficient on {deficient_count} intervals)",
        )
        sizes = np.asarray(active_sizes)
        margins = np.asarray([
            np.min(diagnostics[size]["sigma_min"] / diagnostics[size]["cutoff"])
            for size in sizes
        ])
        accepted = np.asarray([diagnostics[size]["accepted"] for size in sizes])
        axes[1, 1].semilogy(sizes, margins, color="0.45")
        axes[1, 1].scatter(sizes[accepted], margins[accepted], color="tab:blue",
                           marker="o", label="accepted")
        axes[1, 1].scatter(sizes[~accepted], margins[~accepted], color="tab:red",
                           marker="x", label="rejected")
        axes[1, 1].axhline(1.0, color="black", ls=":")
        axes[1, 1].set(
            xlabel=r"Fixed active-space size $N_A$",
            ylabel=r"Trajectory-minimum $\sigma_{\min}/\tau_{\mathrm{clean}}$",
            title="Direct screening of every leading fixed block",
        )
        axes[1, 1].legend()
        filename = "c20_constant_space_selection.png"
    for axis in axes.flat:
        axis.grid(ls="--", alpha=0.28)
    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(figure, filename, output_directory)


def plot_atomistic_energy_blocks(data, active_sizes, constant_rank, output_directory):
    """Plot TiO2 energy branches included in three candidate fixed spaces.

    Parameters
    ----------
    data : dict
        Prepared TiO2 trajectory.
    active_sizes : sequence of three int
        Candidate block dimensions.
    constant_rank : int
        Number of energy branches known to be reliable in the full matrices.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
    for axis, size in zip(axes, active_sizes):
        for state in range(constant_rank):
            included = state < size
            axis.plot(
                data["times_fs"],
                HARTREE_TO_ELECTRONVOLT * data["adiabatic_energies"][:, state],
                color="tab:blue" if included else "0.72",
                lw=2.5 if included else 1.5,
                label=("active" if state == 0 else
                       "excluded" if state == size else None),
            )
        axis.set(xlabel="Time (fs)", ylabel="Adiabatic energy (eV)",
                 title=rf"Candidate fixed space $N_A={size}$")
        axis.grid(ls="--", alpha=0.3)
        axis.legend()
    add_panel_labels(axes)
    figure.tight_layout()
    save_paper_figure(figure, "tio2_constant_space_energies.png", output_directory)


def plot_atomistic_population_comparison(
    tio2_data, tio2_results, tio2_sizes,
    c20_data, c20_results, c20_sizes, c20_diagnostics,
    output_directory,
):
    """Plot selected TiO2 and C20 populations in one paper figure.

    Parameters
    ----------
    tio2_data, c20_data : dict
        Prepared atomistic trajectories.
    tio2_results, c20_results : dict
        Polar-LD propagation results keyed by active-space size.
    tio2_sizes, c20_sizes : sequence of int
        Active-space dimensions displayed for each molecule.
    c20_diagnostics : dict
        C20 support decisions used to label accepted and rejected spaces.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(2, 2, figsize=(19, 14), sharey=True)
    tio2_styles = {
        tio2_sizes[0]: ("tab:red", "--"),
        tio2_sizes[1]: ("tab:orange", "-."),
        tio2_sizes[2]: ("black", "-"),
    }
    for axis, state in zip(axes[0], (5, 4)):
        plot_population_family(
            axis, tio2_data["times_fs"], tio2_results, tio2_sizes,
            state, styles=tio2_styles,
        )
        axis.set(xlabel="Time (fs)", ylabel="Population",
                 title=f"TiO$_2$: adiabatic state {state}",
                 ylim=(-0.03, 1.03))
        axis.grid(ls="--", alpha=0.28)
    axes[0, 0].legend()

    c20_styles = {9: ("tab:orange", "--"), 10: ("black", "-"),
                  14: ("tab:blue", "-."), 41: ("tab:red", ":")}
    for axis, state in zip(axes[1], (8, 7)):
        for size in c20_sizes:
            color, linestyle = c20_styles[size]
            status = "accepted" if c20_diagnostics[size]["accepted"] else "rejected"
            axis.plot(
                c20_data["times_fs"],
                c20_results[size]["reported_population"][:, state],
                color=color, ls=linestyle,
                label=rf"$N_A={size}$ ({status})",
            )
        axis.set(xlabel="Time (fs)", ylabel="Population",
                 title=f"C$_{{20}}$: adiabatic state {state}",
                 ylim=(-0.03, 1.03), xlim=(0, 200))
        axis.grid(ls="--", alpha=0.28)
    axes[1, 0].legend()
    add_panel_labels(axes)
    figure.suptitle("Atomistic unitary LD in selected fixed active spaces")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_paper_figure(figure, "atomistic_ld_populations.png", output_directory)


def _plot_gap_and_support(
    gap_axis, support_axis, energies, sizes, diagnostics, title,
    suggested_boundaries=(),
):
    """Draw one reusable energy-gap/support pair inside the summary figure.

    Parameters
    ----------
    gap_axis, support_axis : matplotlib.axes.Axes
        Destination panels.
    energies : array_like
        Self-consistent adiabatic energies.
    sizes : sequence of int
        Candidate boundary sizes.
    diagnostics : dict
        Support diagnostics for those sizes.
    title : str
        System label prepended to panel titles.
    suggested_boundaries : iterable of int, optional
        Designed band closures highlighted with gold rings.
    """
    gap_data = active_boundary_energy_gaps(energies, sizes)
    sizes = gap_data["active_sizes"]
    minimum = HARTREE_TO_ELECTRONVOLT * np.maximum(
        gap_data["minimum"], 1.0e-12
    )
    accepted = np.asarray([diagnostics[int(size)]["accepted"] for size in sizes])
    margins = np.asarray([
        np.min(diagnostics[int(size)]["sigma_min"] / diagnostics[int(size)]["cutoff"])
        for size in sizes
    ])
    gap_axis.semilogy(sizes, minimum, color="black", lw=4.5,
                      label="minimum boundary gap")
    gap_axis.scatter(sizes[accepted], minimum[accepted], color="tab:blue",
                     marker="o", s=130, zorder=5, label="supported block")
    gap_axis.scatter(sizes[~accepted], minimum[~accepted], color="tab:red",
                     marker="x", s=150, linewidths=4.0, zorder=6,
                     label="unsupported block")
    suggested = np.isin(sizes, suggested_boundaries)
    if np.any(suggested):
        gap_axis.scatter(sizes[suggested], minimum[suggested], facecolors="none",
                         edgecolors="goldenrod", s=280, linewidths=4.5,
                         zorder=7,
                         label="designed band boundary")
    gap_axis.set(xlabel=r"Fixed active-space size $N_A$",
                 ylabel="Energy gap at upper boundary (eV)",
                 title=title + ": energetic candidate boundaries")
    gap_axis.legend()
    support_axis.semilogy(sizes, margins, color="0.42", lw=4.5)
    support_axis.scatter(sizes[accepted], margins[accepted], color="tab:blue",
                         marker="o", s=130, zorder=5, label="supported block")
    support_axis.scatter(sizes[~accepted], margins[~accepted], color="tab:red",
                         marker="x", s=150, linewidths=4.0, zorder=6,
                         label="unsupported block")
    if np.any(suggested):
        support_axis.scatter(sizes[suggested], margins[suggested], facecolors="none",
                             edgecolors="goldenrod", s=280, linewidths=4.5,
                             zorder=7,
                             label="designed band boundary")
    support_axis.axhline(1.0, color="black", ls=":", lw=4.0)
    support_axis.set(xlabel=r"Fixed active-space size $N_A$",
                     ylabel=r"Trajectory-minimum support margin $m_A$",
                     title=title + ": mandatory overlap screening")
    support_axis.legend()
    for axis in (gap_axis, support_axis):
        axis.grid(ls="--", alpha=0.28)


def plot_energy_gap_guidance(cases, output_directory):
    """Plot energy-gap candidate guidance beside mandatory overlap support.

    Parameters
    ----------
    cases : sequence of (energies, sizes, diagnostics, title, boundaries)
        Exactly three physically self-consistent datasets.  Synthetic stress
        filters must not be supplied because their overlaps were altered
        independently of their Hamiltonian energies.
    output_directory : path_like
        Figure destination.
    """
    figure, axes = plt.subplots(3, 2, figsize=(22, 22))
    for row, (energies, sizes, diagnostics, title, boundaries) in enumerate(cases):
        _plot_gap_and_support(axes[row, 0], axes[row, 1], energies, sizes,
                              diagnostics, title, boundaries)
    for axis in axes.flat:
        axis.title.set_fontsize(26)
        axis.xaxis.label.set_fontsize(24)
        axis.yaxis.label.set_fontsize(24)
        axis.tick_params(axis="both", labelsize=21, width=1.8, length=6.5)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(21)
    add_panel_labels(axes, fontsize=26)
    figure.suptitle(
        "Energy gaps propose active-space boundaries; overlaps decide support",
        fontsize=28,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_paper_figure(figure, "energy_gap_candidate_guidance.png", output_directory)
