import numpy as np
import yaml


def linep2dp(x: np.ndarray, y: list | np.ndarray, yn="", title="", xaxis_title="x", yaxis_title="y") -> None:
    """Multi-line 2D plot using plotly.

    :param x: x-axis values.
    :param yv: y-axis values.
    :param yn: y-axis names.
    :param title: Title of the plot.
    :param xaxis_title: Title of the x-axis.
    :param yaxis_title: Title of the y-axis.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=yn))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


if __name__ == "__main__":
    X = np.load("ESR_Continuous_2024-03-07-17-46-03_PCB_ref_50x50.npy")
    X = np.sum(X[0] / X[1], axis=1)

    with open("ESR_Continuous_2024-03-07-17-46-03_PCB_ref_50x50.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        frq = cfg["frequency_values"]  # frequency values in Hz

    print(f"X: {X.shape}")
    print(f"frq: {len(frq)}")

    linep2dp(
        [f * 1e-9 for f in frq],
        X[:, 0, 0],
        title="ESR Spectrum in pixel (0,0)",
        xaxis_title="Frequency [GHz]",
        yaxis_title="Intensity",
    )
