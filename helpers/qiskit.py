import functools
import os.path
import subprocess
import sys

import numpy as np
from qiskit.result import ProbDistribution, QuasiDistribution
from qiskit.visualization import VisualizationError
from qiskit.visualization.counts_visualization import VALID_SORTS, DIST_MEAS, \
    _plot_data
from qiskit.visualization.transition_visualization import _Quaternion, \
    _normalize
from IPython.display import Video
from qiskit.utils import optionals as _optionals
from qiskit.visualization.utils import matplotlib_close_if_inline


def visualize(circuit, trace=True, saveas=None, fpg=200, spg=5):
    """
    Creates animation showing transitions between states of a single
    qubit by applying quantum gates.

    Args:
        circuit (QuantumCircuit): Qiskit single-qubit QuantumCircuit. Gates supported are
            h,x, y, z, rx, ry, rz, s, sdg, t, tdg and u1.
        trace (bool): Controls whether to display tracing vectors - history of 10 past vectors
            at each step of the animation.
        saveas (str): User can choose to save the animation as a video to their filesystem.
            This argument is a string of path with filename and extension (e.g. "movie.mp4" to
            save the video in current working directory).
        fpg (int): Frames per gate. Finer control over animation smoothness and computational
            needs to render the animation. Works well for tkinter GUI as it is, for jupyter GUI
            it might be preferable to choose fpg between 5-30.
        spg (int): Seconds per gate. How many seconds should animation of individual gate
            transitions take.

    Returns:
        IPython.core.display.HTML:
            If arg jupyter is set to True. Otherwise opens tkinter GUI and returns
            after the GUI is closed.

    Raises:
        MissingOptionalLibraryError: Must have Matplotlib (and/or IPython) installed.
        VisualizationError: Given gate(s) are not supported.

    """
    try:
        from IPython.display import HTML

        has_ipython = True
    except ImportError:
        has_ipython = False

    try:
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D
        from qiskit.visualization.bloch import Bloch
        from qiskit.visualization.exceptions import VisualizationError

        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    saveas = "media/" + saveas if saveas else None
    converted_path = saveas + '.webm'
    if os.path.exists(converted_path):
        return Video(converted_path, embed=True, width=500, height=500)

    jupyter = False
    if ("ipykernel" in sys.modules) and ("spyder" not in sys.modules):
        jupyter = True

    if not has_matplotlib:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="visualize_transition",
            pip_install="pip install matplotlib",
        )
    if not has_ipython and jupyter is True:
        raise MissingOptionalLibraryError(
            libname="IPython",
            name="visualize_transition",
            pip_install="pip install ipython",
        )
    if len(circuit.qubits) != 1:
        raise VisualizationError("Only one qubit circuits are supported")

    frames_per_gate = fpg
    time_between_frames = (spg * 1000) / fpg

    # quaternions of gates which don't take parameters
    simple_gates = {}
    simple_gates["x"] = (
        "x",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [1, 0, 0]),
        "#1abc9c",
    )
    simple_gates["y"] = (
        "y",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 1, 0]),
        "#2ecc71",
    )
    simple_gates["z"] = (
        "z",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 0, 1]),
        "#3498db",
    )
    simple_gates["s"] = (
        "s",
        _Quaternion.from_axisangle(np.pi / 2 / frames_per_gate, [0, 0, 1]),
        "#9b59b6",
    )
    simple_gates["sdg"] = (
        "sdg",
        _Quaternion.from_axisangle(-np.pi / 2 / frames_per_gate, [0, 0, 1]),
        "#8e44ad",
    )
    simple_gates["h"] = (
        "h",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, _normalize([1, 0, 1])),
        "#34495e",
    )
    simple_gates["t"] = (
        "t",
        _Quaternion.from_axisangle(np.pi / 4 / frames_per_gate, [0, 0, 1]),
        "#e74c3c",
    )
    simple_gates["tdg"] = (
        "tdg",
        _Quaternion.from_axisangle(-np.pi / 4 / frames_per_gate, [0, 0, 1]),
        "#c0392b",
    )

    list_of_circuit_gates = []

    for gate, _, _ in circuit._data:
        if gate.name == "barrier":
            continue
        if gate.name in simple_gates:
            list_of_circuit_gates.append(simple_gates[gate.name])
        elif gate.name == "rx":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [1, 0, 0])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#16a085"))
        elif gate.name == "ry":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 1, 0])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#27ae60"))
        elif gate.name == "rz":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 0, 1])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#2980b9"))
        elif gate.name == "u1":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 0, 1])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#f1c40f"))
        # Commented to skip unsupported gates silently, like reset
        # else:
        #     raise VisualizationError(f"Gate {gate.name} is not supported")

    if len(list_of_circuit_gates) == 0:
        raise VisualizationError("Nothing to visualize.")

    starting_pos = _normalize(np.array([0, 0, 1]))

    fig = plt.figure(figsize=(5, 5))
    if tuple(int(x) for x in matplotlib.__version__.split(".")) >= (3, 4, 0):
        _ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(_ax)
    else:
        _ax = Axes3D(fig)

    _ax.set_xlim(-10, 10)
    _ax.set_ylim(-10, 10)
    sphere = Bloch(axes=_ax)

    class Namespace:
        """Helper class serving as scope container"""

        def __init__(self):
            self.new_vec = []
            self.last_gate = -2
            self.colors = []
            self.pnts = []

    namespace = Namespace()
    namespace.new_vec = starting_pos

    def animate(i):
        sphere.clear()

        # starting gate count from -1 which is the initial vector
        gate_counter = (i - 1) // frames_per_gate
        if gate_counter != namespace.last_gate:
            namespace.pnts.append([[], [], []])
            namespace.colors.append(list_of_circuit_gates[gate_counter][2])

        # starts with default vector [0,0,1]
        if i == 0:
            sphere.add_vectors(namespace.new_vec)
            namespace.pnts[0][0].append(namespace.new_vec[0])
            namespace.pnts[0][1].append(namespace.new_vec[1])
            namespace.pnts[0][2].append(namespace.new_vec[2])
            namespace.colors[0] = "r"
            sphere.make_sphere()
            return _ax

        namespace.new_vec = list_of_circuit_gates[gate_counter][1] * namespace.new_vec

        namespace.pnts[gate_counter + 1][0].append(namespace.new_vec[0])
        namespace.pnts[gate_counter + 1][1].append(namespace.new_vec[1])
        namespace.pnts[gate_counter + 1][2].append(namespace.new_vec[2])

        sphere.add_vectors(namespace.new_vec)
        if trace:
            # sphere.add_vectors(namespace.points)
            for point_set in namespace.pnts:
                sphere.add_points([point_set[0], point_set[1], point_set[2]])

        sphere.vector_color = [list_of_circuit_gates[gate_counter][2]]
        sphere.point_color = namespace.colors
        sphere.point_marker = "o"

        annotation_text = list_of_circuit_gates[gate_counter][0]
        annotationvector = [1.40, -0.45, 1.65]
        sphere.add_annotation(
            annotationvector,
            annotation_text,
            color=list_of_circuit_gates[gate_counter][2],
            fontsize=30,
            horizontalalignment="left",
        )

        sphere.make_sphere()

        namespace.last_gate = gate_counter
        return _ax

    def init():
        sphere.vector_color = ["r"]
        return _ax

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(frames_per_gate * len(list_of_circuit_gates) + 1),
        init_func=init,
        blit=False,
        repeat=False,
        interval=time_between_frames,
    )

    if saveas:
        saveas_path = saveas + '.mp4'
        ani.save(saveas_path, fps=60, dpi=200, bitrate=1000000)
        if jupyter:
            cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-i", saveas_path,
                "-c:v", "libvpx-vp9",
                "-b:v", "0",
                "-crf", "30",
                "-c:a", "libopus",
                "-pix_fmt", "yuv420p",
                converted_path,
            ]
            subprocess.run(cmd, check=True)
            plt.close(fig)
            return Video(converted_path, embed=True, width=500, height=500)

    if jupyter:
        # This is necessary to overcome matplotlib memory limit
        matplotlib.rcParams["animation.embed_limit"] = 50
        plt.close(fig)
        # return HTML(ani.to_html5_video())
        html_str = ani.to_jshtml()
        # with open(path, "w", encoding="utf-8") as f:
        #     f.write(html_str)
        return HTML(html_str)
    plt.show()
    plt.close(fig)
    return None


def plot_histogram(
    data,
    figsize=None,
    color=None,
    number_to_keep=None,
    sort="asc",
    target_string=None,
    legend=None,
    bar_labels=True,
    title=None,
    ax=None,
    filename=None,
):
    """Plot a histogram of input counts data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex ``{'001': 130}``)

        figsize (tuple): Figure size in inches.
        color (list or str): String or list of strings for histogram bar colors.
        number_to_keep (int): The number of terms to plot per dataset.  The rest is made into a
            single bar called 'rest'.  If multiple datasets are given, the ``number_to_keep``
            applies to each dataset individually, which may result in more bars than
            ``number_to_keep + 1``.  The ``number_to_keep`` applies to the total values, rather than
            the x-axis sort.
        sort (string): Could be `'asc'`, `'desc'`, `'hamming'`, `'value'`, or
            `'value_desc'`. If set to `'value'` or `'value_desc'` the x axis
            will be sorted by the number of counts for each bitstring.
            Defaults to `'asc'`.
        target_string (str): Target string if 'sort' is a distance measure.
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the length of data (if data is a
            list or 1 if it's a dict)
        bar_labels (bool): Label each bar in histogram with counts value.
        title (str): A string to use for the plot title
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        filename (str): file path to save image to.

    Returns:
        matplotlib.Figure:
            A figure for the rendered histogram, if the ``ax``
            kwarg is not set.

    Raises:
        MissingOptionalLibraryError: Matplotlib not available.
        VisualizationError: When legend is provided and the length doesn't
            match the input data.
        VisualizationError: Input must be Counts or a dict

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            # Plot two counts in the same figure with legends and colors specified.

            from qiskit.visualization import plot_histogram

            counts1 = {'00': 525, '11': 499}
            counts2 = {'00': 511, '11': 514}

            legend = ['First execution', 'Second execution']

            plot_histogram([counts1, counts2], legend=legend, color=['crimson','midnightblue'],
                            title="New Histogram")

            # You can sort the bitstrings using different methods.

            counts = {'001': 596, '011': 211, '010': 50, '000': 117, '101': 33, '111': 8,
                    '100': 6, '110': 3}

            # Sort by the counts in descending order
            hist1 = plot_histogram(counts, sort='value_desc')

            # Sort by the hamming distance (the number of bit flips to change from
            # one bitstring to the other) from a target string.
            hist2 = plot_histogram(counts, sort='hamming', target_string='001')
    """
    if not isinstance(data, list):
        data = [data]

    kind = "counts"
    for dat in data:
        if isinstance(dat, (QuasiDistribution, ProbDistribution)) or isinstance(
            next(iter(dat.values())), float
        ):
            kind = "distribution"
    return _plotting_core(
        data,
        figsize,
        color,
        number_to_keep,
        sort,
        target_string,
        legend,
        bar_labels,
        title,
        ax,
        filename,
        kind=kind,
    )


def plot_distribution(
    data,
    figsize=(7, 5),
    color=None,
    number_to_keep=None,
    sort="asc",
    target_string=None,
    legend=None,
    bar_labels=True,
    title=None,
    ax=None,
    filename=None,
):
    """Plot a distribution from input sampled data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        figsize (tuple): Figure size in inches.
        color (list or str): String or list of strings for distribution bar colors.
        number_to_keep (int): The number of terms to plot per dataset.  The rest is made into a
            single bar called 'rest'.  If multiple datasets are given, the ``number_to_keep``
            applies to each dataset individually, which may result in more bars than
            ``number_to_keep + 1``.  The ``number_to_keep`` applies to the total values, rather than
            the x-axis sort.
        sort (string): Could be `'asc'`, `'desc'`, `'hamming'`, `'value'`, or
            `'value_desc'`. If set to `'value'` or `'value_desc'` the x axis
            will be sorted by the maximum probability for each bitstring.
            Defaults to `'asc'`.
        target_string (str): Target string if 'sort' is a distance measure.
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the length of data (if data is a
            list or 1 if it's a dict)
        bar_labels (bool): Label each bar in histogram with probability value.
        title (str): A string to use for the plot title
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        filename (str): file path to save image to.

    Returns:
        matplotlib.Figure:
            A figure for the rendered distribution, if the ``ax``
            kwarg is not set.

    Raises:
        MissingOptionalLibraryError: Matplotlib not available.
        VisualizationError: When legend is provided and the length doesn't
            match the input data.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            # Plot two counts in the same figure with legends and colors specified.

            from qiskit.visualization import plot_distribution

            counts1 = {'00': 525, '11': 499}
            counts2 = {'00': 511, '11': 514}

            legend = ['First execution', 'Second execution']

            plot_distribution([counts1, counts2], legend=legend, color=['crimson','midnightblue'],
                            title="New Distribution")

            # You can sort the bitstrings using different methods.

            counts = {'001': 596, '011': 211, '010': 50, '000': 117, '101': 33, '111': 8,
                    '100': 6, '110': 3}

            # Sort by the counts in descending order
            dist1 = plot_distribution(counts, sort='value_desc')

            # Sort by the hamming distance (the number of bit flips to change from
            # one bitstring to the other) from a target string.
            dist2 = plot_distribution(counts, sort='hamming', target_string='001')

    """
    return _plotting_core(
        data,
        figsize,
        color,
        number_to_keep,
        sort,
        target_string,
        legend,
        bar_labels,
        title,
        ax,
        filename,
        kind="distribution",
    )


@_optionals.HAS_MATPLOTLIB.require_in_call
def _plotting_core(
    data,
    figsize=(7, 5),
    color=None,
    number_to_keep=None,
    sort="asc",
    target_string=None,
    legend=None,
    bar_labels=True,
    title=None,
    ax=None,
    filename=None,
    kind="counts",
):
    fontsize = 16
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if sort not in VALID_SORTS:
        raise VisualizationError(
            "Value of sort option, %s, isn't a "
            "valid choice. Must be 'asc', "
            "'desc', 'hamming', 'value', 'value_desc'"
        )
    if sort in DIST_MEAS and target_string is None:
        err_msg = "Must define target_string when using distance measure."
        raise VisualizationError(err_msg)

    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError(
            f"Length of legend ({len(legend)}) doesn't match number of input executions ({len(data)})."
        )

    # Set bar colors
    if color is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    elif isinstance(color, str):
        color = [color]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    labels = sorted(functools.reduce(lambda x, y: x.union(y.keys()), data, set()))
    if number_to_keep is not None:
        labels.append("rest")

    if sort in DIST_MEAS:
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string) if item != "rest" else 0)

        labels = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][1]
    elif "value" in sort:
        combined_counts = {}
        if isinstance(data, dict):
            combined_counts = data
        else:
            for counts in data:
                for count in counts:
                    prev_count = combined_counts.get(count, 0)
                    combined_counts[count] = max(prev_count, counts[count])
        labels = sorted(combined_counts.keys(), key=lambda key: combined_counts[key])

    length = len(data)
    width = 1 / (len(data) + 1)  # the width of the bars

    labels_dict, all_pvalues, all_inds = _plot_data(data, labels, number_to_keep, kind=kind)
    rects = []
    for item, _ in enumerate(data):
        label = None
        for idx, val in enumerate(all_pvalues[item]):
            if not idx and legend:
                label = legend[item]
            if val > 0:
                rects.append(
                    ax.bar(
                        idx + item * width,
                        val,
                        width,
                        label=label,
                        color=color[item % len(color)],
                        zorder=2,
                    )
                )
                label = None
        bar_center = (width / 2) * (length - 1)
        ax.set_xticks(all_inds[item] + bar_center)
        ax.set_xticklabels(labels_dict.keys(), rotation=70, ha="right", rotation_mode="anchor")
        # attach some text labels
        if bar_labels:
            for rect in rects:
                for rec in rect:
                    height = rec.get_height()
                    if kind == "distribution":
                        height = round(height, 3)
                    if height >= 1e-3:
                        ax.text(
                            rec.get_x() + rec.get_width() / 2.0,
                            1.05 * height,
                            str(height),
                            ha="center",
                            va="bottom",
                            zorder=3,
                            fontsize=fontsize,
                        )
                    else:
                        ax.text(
                            rec.get_x() + rec.get_width() / 2.0,
                            1.05 * height,
                            "0",
                            ha="center",
                            va="bottom",
                            zorder=3,
                            fontsize=fontsize,
                        )
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    # add some text for labels, title, and axes ticks
    if kind == "counts":
        ax.set_ylabel("Count", fontsize=fontsize)
    else:
        ax.set_ylabel("Quasi-probability", fontsize=fontsize)
    all_vals = np.concatenate(all_pvalues).ravel()
    min_ylim = 0.0
    if kind == "distribution":
        min_ylim = min(0.0, min(1.1 * val for val in all_vals))
    ax.set_ylim([min_ylim, min(
        [1.2 * sum(all_vals), max(1.2 * val for val in all_vals)])])
    if "desc" in sort:
        ax.invert_xaxis()

    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.grid(which="major", axis="y", zorder=0, linestyle="--")
    if title:
        plt.title(title, fontsize=fontsize)

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            borderaxespad=0,
            frameon=True,
        )
    if fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)