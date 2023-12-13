"""This module contains functions for plotting simulated CSDs.

@author: f.hader
"""

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def plot_csd(
    csd: np.ndarray,
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    title: Optional[str] = None,
    sensor_label: str = "Sensor signal [a.u.]",
    gate_labels: Optional[list] = None,
    voltage_unit: str = "V",
    save_path: Optional[str] = None,
    sort_volts: bool = True,
    num_xticks: int = 5,
    num_yticks: int = 5,
) -> None:
    """Plots a CSD. Can be used for 2D and 1D scans.

    Args:
        csd (np.ndarray): Numpy array containing the CSD. For 2D scans (csd.ndim == 2): First-/y-axis = (plunger) gate
            2, second-/x-axis = (plunger) gate 1. Both voltages are expected to map to the axes taking into account the
            direction of the measurement represented in the volt_limits.
        volt_limits_g1 (np.ndarray): Voltage limits of (plunger) gate 1 (second-/x-axis). \n
            Example: \n
            [start_V1, stop_V1]
        volt_limits_g2 (np.ndarray): Voltage limits of (plunger) gate 2 (first-/y-axis). \n
            Example: \n
            [start_V2, stop_V2]
        title (Optional[str]): The title of the plot. Default is None.
        sensor_label (str): The label for the sensor value (2D scan: colorbar label, 1D scan: y-axis-label). Default is
            'Sensor signal [a.u.]'.
        gate_labels (Optional[list]): The labels of the gates with voltages volt_limits_g1 respective volt_limits_g2.
        voltage_unit (str): The unit for the volt_limits.
        save_path (Optional[str]): Filepath at which the plot will be saved. Default is None.
        sort_volts (bool): Specifies if the plot should show the lowest voltages at the lowest index (and if required
            flip the image). For 1D data it is sorted by gate 1. Else the voltages will appear as defined by the
            volt_limits. Default is True.
        num_xticks (int): Specifies how many xticks are displayed. Default is 5.
        num_yticks (int): Specifies how many yticks are displayed. Default is 5.
    """
    if gate_labels is None:
        gate_labels = ["g1", "g2"]
    if csd.ndim == 1:
        _plot_csd_1d(
            csd=csd,
            volt_limits_g1=volt_limits_g1,
            volt_limits_g2=volt_limits_g2,
            title=title,
            y_label=sensor_label,
            x_label=f"({gate_labels[0]};{gate_labels[1]}) [{voltage_unit}]",
            save_path=save_path,
            sort_volts=sort_volts,
            num_xticks=num_xticks,
            num_yticks=num_yticks,
        )
    elif csd.ndim == 2:
        _plot_csd_2d(
            csd=csd,
            volt_limits_g1=volt_limits_g1,
            volt_limits_g2=volt_limits_g2,
            title=title,
            cb_label=sensor_label,
            x_label=f"{gate_labels[0]} [{voltage_unit}]",
            y_label=f"{gate_labels[1]} [{voltage_unit}]",
            save_path=save_path,
            sort_volts=sort_volts,
            num_xticks=num_xticks,
            num_yticks=num_yticks,
        )


def _plot_csd_2d(
    csd: np.ndarray,
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    title: Optional[str] = None,
    cb_label: str = "Sensor signal [a.u.]",
    x_label: str = "g1 [V]",
    y_label: str = "g2 [V]",
    save_path: Optional[str] = None,
    sort_volts: bool = True,
    num_xticks: int = 5,
    num_yticks: int = 5,
) -> None:
    """Plots a 2D CSD.

    Args:
        csd (np.ndarray): Two dimensional numpy array containing the CSD. First-/y-axis = (plunger) gate 2,
            second-/x-axis = (plunger) gate 1. Both voltages are expected to map to the axes taking into account the
            direction of the measurement represented in the volt_limits.
        volt_limits_g1 (ndarray): Voltage limits of (plunger) gate 1 (second-/x-axis). \n
            Example: \n
            [start_V1, stop_V1]
        volt_limits_g2 (np.ndarray): Voltage limits of (plunger) gate 2 (first-/y-axis). \n
            Example: \n
            [start_V2, stop_V2]
        title (Optional[str]): The title of the plot. Default is None.
        cb_label (str): The label of the colorbar. Default is 'Sensor signal [a.u.]'. If it is set to `None`, the
            colorbar is not plotted. For plotting the colorbar without a label pass "".
        x_label (str): The label of the x-axis. Default is 'g1 [V]'.
        y_label (str): The label of the y-axis. Default is 'g2 [V]'.
        save_path (Optional[str]): Filepath at which the plot will be saved. Default is None.
        sort_volts (bool): Specifies if the plot should show the lowest voltages at the lowest index (and if required
            flip the image). Else the voltages will appear as defined by the volt_limits. Default is True.
        num_xticks (int): Specifies how many xticks are displayed. Default is 5.
        num_yticks (int): Specifies how many yticks are displayed. Default is 5.
    """
    # sort CSD so that lowest voltages are at index 0 for both gates
    if sort_volts:
        if volt_limits_g2[0] > volt_limits_g2[1]:
            csd = csd[::-1, :]
            volt_limits_g2 = volt_limits_g2[::-1]
        if volt_limits_g1[0] > volt_limits_g1[1]:
            csd = csd[:, ::-1]
            volt_limits_g1 = volt_limits_g1[::-1]

    x_extent = np.max(volt_limits_g1) - np.min(volt_limits_g1)
    y_extent = np.max(volt_limits_g2) - np.min(volt_limits_g2)
    aspect = (y_extent / csd.shape[0]) / (x_extent / csd.shape[1])
    plt.imshow(csd, aspect=aspect, origin="lower", interpolation="none")
    plt.xticks(
        np.linspace(0, csd.shape[1] - 1, num_xticks),
        ["%.3f" % x for x in np.linspace(volt_limits_g1[0], volt_limits_g1[1], num_xticks).round(3)],
    )
    plt.yticks(
        np.linspace(0, csd.shape[0] - 1, num_yticks),
        ["%.3f" % x for x in np.linspace(volt_limits_g2[0], volt_limits_g2[1], num_yticks).round(3)],
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if cb_label is not None:
        cb = plt.colorbar()
        cb.set_label(label=cb_label, fontsize=12)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def _plot_csd_1d(
    csd: np.ndarray,
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    title: Optional[str] = None,
    x_label: str = "(g1;g2) [V]",
    y_label: str = "Sensor signal [a.u.]",
    save_path: Optional[str] = None,
    sort_volts: bool = True,
    num_xticks: int = 5,
    num_yticks: int = 5,
) -> None:
    """Plots a 1D CSD.

    Args:
        csd (np.ndarray): One dimensional numpy array containing the CSD. Both voltages are expected to map to the
            axis taking into account the direction of the measurement represented in the volt_limits.
        volt_limits_g1 (ndarray): Voltage limits of plunger gate 1 (second-/x-axis). \n
            Example: \n
            [start_V1, stop_V1]
        volt_limits_g2 (np.ndarray): Voltage limits of plunger gate 2 (first-/y-axis). \n
            Example: \n
            [start_V2, stop_V2]
        title (Optional[str]): The title of the plot. Default is None.
        y_label (str): The label of the y-axis. Default is 'Sensor signal [a.u.]'.
        save_path (Optional[str]): Filepath at which the plot will be saved. Default is None.
        sort_volts (bool): Specifies if the plot should show the lowest voltages at the lowest index (and if required
            flip the image). For 1D data it is sorted by gate 1. Else the voltages will appear as defined by the
            volt_limits. Default is True.
        num_xticks (int): Specifies how many xticks are displayed. Default is 5.
        num_yticks (int): Specifies how many yticks are displayed. Default is 5.
    """
    # sort CSD so that lowest voltage of gate 1 is at index 0
    if sort_volts and volt_limits_g1[0] > volt_limits_g1[1]:
        csd = csd[::-1]
        volt_limits_g1 = volt_limits_g1[::-1]
        volt_limits_g2 = volt_limits_g2[::-1]

    plt.plot(csd)
    # prepare x_ticks
    # g1 voltages
    g1_voltages = np.linspace(volt_limits_g1[0], volt_limits_g1[1], num_xticks).round(3)
    # g2 voltages
    g2_voltages = np.linspace(volt_limits_g2[0], volt_limits_g2[1], num_xticks).round(3)
    x_ticks = [f"({g1_voltages[i]:.3f};{g2_voltages[i]:.3f})" for i in range(g1_voltages.size)]
    plt.xticks(np.linspace(0, csd.shape[0] - 1, num_xticks), x_ticks)
    plt.yticks(np.linspace(np.min(csd), np.max(csd), num_yticks))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
