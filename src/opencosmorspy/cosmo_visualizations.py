import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
import rdkit.Chem.rdDetermineBonds

from opencosmorspy.input_parsers import SigmaProfileParser

def get_atom_color_map(atoms_available=None):
    element_color_discrete_map = {
        # Blended H-bonded atoms (50% white + atom color)
        "H-C": "#E4E4E4",    # Light gray (between white and C gray)
        "H-O": "#FF8A8A",    # Light red
        "H-N": "#98A8FC",    # Light blue
        "H-S": "#FFE88F",    # Light yellow
        "H-F": "#C8F4A8",    # Pale green
        "H-Cl": "#BFFFBF",   # Mint green
        "H-Br": "#D59494",   # Dusty pink
        "H-I": "#EAC0EA",    # Soft purple
        "H-P": "#FFD1A0",    # Light orange
        "H-B": "#FFDADA",    # Blush pink
        "H-Si": "#F1DCA0",   # Light tan

        # Standard atoms (CPK colors)
        "H": "#FFFFFF",      # White
        "C": "#909090",      # Gray
        "N": "#3050F8",      # Blue
        "O": "#FF0D0D",      # Red
        "F": "#90E050",      # Green
        "Cl": "#1FF01F",     # Bright green
        "Br": "#A62929",     # Brownish red
        "I": "#940094",      # Violet
        "S": "#FFD123",      # Yellow
        "P": "#FF8000",      # Orange
        "B": "#FFB5B5",      # Pink
        "Si": "#DAA520",     # Goldenrod/tan
        "Na": "#0000FF",     # Blue
        "K": "#8F40D4",      # Purple
        "Mg": "#8AFF00",     # Lime
        "Ca": "#3DFF00",     # Green
        "Fe": "#E06633",     # Rusty orange
        "Zn": "#7D80B0",     # Light steel blue
    }
    missing_labels = [AN for AN in atoms_available if AN not in element_color_discrete_map]
    safe_palette = px.colors.qualitative.Safe
    for i, label in enumerate(missing_labels):
        color = safe_palette[i % len(safe_palette)]
        element_color_discrete_map[label] = color

    return element_color_discrete_map


def plot_sigma_profiles(
    filepath_lst,
    plot_name=None,
    plot_label_dct=None,
    xlim=(-0.02, 0.02),
    dir_plot=None,
    aggregate_plots=False
):
    if not dir_plot:
        dir_plot = os.getcwd()

    if plot_name is None:
        plot_name = "sigma_profiles"

    plot_label_lst = []
    for filepath in filepath_lst:
        if plot_label_dct is not None and filepath in plot_label_dct:
            plot_label_lst.append(plot_label_dct[filepath])
        else:
            plot_label_lst.append(os.path.basename(filepath))

    fig, ax = plt.subplots(figsize=(12, 6))

    for filepath, label in zip(filepath_lst, plot_label_lst):
        spp = SigmaProfileParser(filepath)
        sigmas, areas = spp.cluster_and_create_sigma_profile()
        ax.plot(sigmas, areas, label=label)

        ax.set_xlim(*xlim)

        if aggregate_plots:
            if dir_plot:
                fig.savefig(os.path.join(dir_plot,  f"{plot_name}_{label}.png"), dpi=300)
            else:
                plt.show()

    if not aggregate_plots:
        if dir_plot:
            fig.savefig(os.path.join(dir_plot,  f"{plot_name}_{label}.png"), dpi=300)
        else:
            plt.show()


def plot_sigma_profiles_plotly(
    filepath_lst,
    plot_name=None,
    plot_label_dct=None,
    xlim=(-0.02, 0.02),
    dir_plot=None,
    mode="static"
):
    if not dir_plot:
        dir_plot = os.getcwd()

    if plot_name is None:
        plot_name = "sigma_profiles"

    plot_label_lst = []
    for filepath in filepath_lst:
        if plot_label_dct is not None and filepath in plot_label_dct:
            plot_label_lst.append(plot_label_dct[filepath])
        else:
            plot_label_lst.append(os.path.basename(filepath))

    fontsize = 20

    if mode == "dynamic":
        xaxis_title = "sigma"
        yaxis_title = "p(sigma)"
    elif mode == "static":
        xaxis_title = "$\large \sigma \; [e/ 10^{-10} m]$"
        yaxis_title = "$\large p(\sigma) \; [-]$"

    fig = go.Figure()

    y_max = 0
    for idx, (filepath, label) in enumerate(
        zip(filepath_lst, plot_label_lst)
    ):

        if idx == 0:
            line = {"dash": "solid", "color": "black"}  # blue '#316395'
        elif idx == 1:
            line = {"dash": "longdash", "color": "#AF0000"}
        elif idx == 2:
            line = {"dash": "dashdot", "color": "#109618"}
        elif idx == 4:
            line = {"dash": "dot", "color": "#09118C"}
        elif idx == 5:
            line = {"dash": "dash", "color": "#7600b5"}

        elif idx == 5:
            line = {"dash": "longdashdot", "color": "#DEBC00"}
        elif idx == 6:
            line = {"dash": "dot", "color": "#565656"}

        spp = SigmaProfileParser(filepath)
        sigmas, areas = spp.cluster_and_create_sigma_profile()
        y_max = max(y_max, areas.max())

        basename = os.path.basename(filepath)

        fig.add_trace(
            go.Scatter(
                x=sigmas,
                y=areas,
                mode="lines",
                name=basename,
                line=line,
            )
        )

    fig.update_layout(
        width=700,
        height=500,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        # margin=dict(
        #     l=50,
        #     r=50,
        #     b=50,
        #     t=50,
        #     pad=4),
    )
    fig.update_layout(
        {
            "xaxis_range": xlim,
            "yaxis_range": [0, y_max * 1.2],
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "White",
            "font": {"size": fontsize, "color": "rgba(1, 1, 1, 1)"},
            "legend_title_text": "",
            "showlegend": False,
            "legend": {
                "yanchor": "top",
                "y": 0.98,
                "xanchor": "right",
                "x": 0.98,
                "bgcolor": "White",
                "bordercolor": "Black",
                "borderwidth": 0.5,
            },
        }
    )

    fig.update_xaxes(
        tick0=-0.02,
        dtick=0.005,
        color="rgba(1, 1, 1, 1)",
        gridcolor="rgba(0, 0, 0, 0.0)",
        showgrid=False,
        zeroline=False,
        zerolinecolor="rgba(1, 1, 1, 1)",
        zerolinewidth=0.5,
        tickformat=".3f",
        mirror=True,
        linecolor="black",
        ticks="outside",
        showline=True,
    )
    fig.update_yaxes(
        tick0=0,
        # dtick = 1,
        color="rgba(1, 1, 1, 1)",
        gridcolor="rgba(0, 0, 0, 0.0)",
        showgrid=False,
        zeroline=True,
        zerolinecolor="rgba(1, 1, 1, 1)",
        zerolinewidth=0.5,
        tickformat=".1f",
        mirror=True,
        linecolor="black",
        ticks="outside",
        showline=True,
    )

    fig.show()
    if mode == "static":
        fig.write_image(os.path.join(dir_plot, plot_name + ".png"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".pdf"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".svg"))
    if mode == "dynamic":
        fig.write_html(os.path.join(dir_plot, plot_name + ".html"))


def plot_extended_sigma_profile_plotly(filepath, area_max_size=30, dir_plot=None, mode="dynamic"):
    if not dir_plot:
        dir_plot = os.getcwd()
    plot_name = "extsp_" + os.path.splitext(os.path.basename(filepath))[0]
    
    spp = SigmaProfileParser(filepath)
    spp.calculate_averaged_sigmas(averaging_radius=1)
    sigmas_corr = spp['seg_sigma_averaged'].copy()
    spp.calculate_averaged_sigmas()
    sigmas = spp['seg_sigma_averaged']
    
    sigma_orth = sigmas_corr - 0.816 * sigmas
    pt = Chem.GetPeriodicTable()
    atom_AN = [pt.GetAtomicNumber(v) for v in spp['atm_elmnt']]
    for i, AN in enumerate(atom_AN):
        if AN == 1:
            adjacent_index = np.flatnonzero(spp['adjacency_matrix'][i, :])[0]
            atom_AN[i] = 100 + atom_AN[adjacent_index]
    atom_AN = np.array(atom_AN)
    seg_AN = np.array([atom_AN[n] for n in spp['seg_atm_nr']])

    descriptors = [sigmas, sigma_orth, seg_AN]
    descriptor_ranges = [np.arange(-0.03, 0.03, 0.001), np.arange(-0.03, 0.03, 0.001), np.sort(np.unique(atom_AN))]
    clustered_descriptors, clustered_areas = spp.cluster_segments_into_segmenttypes(
        descriptors, descriptor_ranges
    )
    data = {
        'sigma': clustered_descriptors[:, 0].tolist(),
        'sigma_orth': clustered_descriptors[:, 1].tolist(),
        'seg_AN': clustered_descriptors[:, 2].tolist(),
        'area': clustered_areas.tolist(),
    }
    df_esp = pd.DataFrame(data)

    ANs = []
    AN_labels = []
    for AN in sorted(set(atom_AN)):
        if AN > 100:
            ANs.append(AN)
            AN_labels.append(f'H-{pt.GetElementSymbol(int(AN - 100))}')
    for AN in sorted(set(atom_AN)):
        if AN < 100:
            ANs.append(AN)
            AN_labels.append(pt.GetElementSymbol(int(AN)))

    df_esp["seg_AN_label"] = ''
    for AN, AN_label in zip(ANs, AN_labels):
        df_esp.loc[df_esp["seg_AN"] == AN, "seg_AN_label"] = AN_label

    # Make small areas visible, as they are clusters
    df_esp.loc[(df_esp["area"] != 0) & (df_esp["area"] < 0.06), "area"] = 0.06

    # Pivot
    df_temp = pd.pivot_table(df_esp, values="sigma_orth", index="sigma", aggfunc='sum')
    df_temp.reset_index(inplace=True)

    fontsize = 24
    fontsize_tick = 24

    # In static mode latex works very bad
    if mode == "static":
        labels = {
            "sigma": r"$\Large\sigma [e/10^{-10} m]$",
            "sigma_orth": r"$\Large\sigma^{\perp} [e/10^{-10} m]$",
        }
    elif mode == "dynamic":
        labels = {}
    
    atom_color_map = get_atom_color_map(AN_labels)
    fig = px.scatter(
        df_esp,
        x="sigma",
        y="sigma_orth",
        color="seg_AN_label",
        labels=labels,
        size="area",
        hover_data=["sigma", "sigma_orth", "seg_AN"],
        size_max=area_max_size,
        opacity=0.7,
        color_discrete_map=atom_color_map,
        category_orders={"seg_AN_label": AN_labels},
        width=800,
        height=500,
    )
    fig.update_layout(
        {
            "xaxis_range": [-0.025, 0.025],
            "yaxis_range": [-0.007, 0.007],
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "White",
            "font": {"size": fontsize, "color": "rgba(1, 1, 1, 1)"},
            "legend_title_text": "",
            "legend": {
                "yanchor": "top",
                "y": 0.98,
                "xanchor": "left",
                "x": 0.85,
                "bgcolor": "White",
                "bordercolor": "Black",
                "borderwidth": 0.5,
            },
        }
    )

    fig.update_xaxes(
        tick0=-0.020,
        tickfont={"size": fontsize_tick},
        dtick=0.01,
        color="rgba(1, 1, 1, 1)",
        gridcolor="rgba(0, 0, 0, 0.0)",
        zeroline=False,
        tickformat=".3f",
        mirror=True,
        linecolor="black",
        ticks="outside",
        showline=True,
    )
    fig.update_yaxes(
        tick0=-0.006,
        dtick=0.002,
        tickfont={"size": fontsize_tick},
        color="rgba(1, 1, 1, 1)",
        gridcolor="rgba(0, 0, 0, 0.0)",
        zeroline=False,
        tickformat=".3f",
        mirror=True,
        linecolor="black",
        ticks="outside",
        showline=True,
    )

    fig.show()
    if mode == "static":
        fig.write_image(os.path.join(dir_plot, plot_name + ".png"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".pdf"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".svg"))
    if mode == "dynamic":
        fig.write_html(os.path.join(dir_plot, plot_name + ".html"))


def plot_3D_segment_location(filepath, mode="dynamic", dir_plot=".", plot_name="cosmo_surface"):

    spp = SigmaProfileParser(filepath)
    mol = rdkit.Chem.MolFromXYZBlock(spp.save_to_xyz())
    charge = int(spp['seg_charge'].sum())
    rdkit.Chem.rdDetermineBonds.DetermineBonds(mol, charge=charge)
    if mol is None:
        raise ValueError("Unable to load molecule from XYZ file.")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=spp["seg_pos"][:, 0],
                y=spp["seg_pos"][:, 1],
                z=spp["seg_pos"][:, 2],
                mode="markers",
                marker=dict(
                    size=spp["seg_area"]
                    * 50,  # Scale the size for better visualization
                    color=spp["seg_charge"],  # Set the color based on charge
                    colorscale="bluered",  # Color scale for charges
                    colorbar=dict(title="Charge"),
                    opacity=0.7,
                ),
                name=None,
                showlegend=False,
            )
        ]
    )
    atoms_available = sorted(set(spp['atm_elmnt']))
    atom_color_map = get_atom_color_map(atoms_available)

    # Add bonds as lines (RDKit automatically infers bonds from 3D structure)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add bond (line) between the two atoms
        fig.add_trace(
            go.Scatter3d(
                x=[spp["atm_pos"][i, 0], spp["atm_pos"][j, 0]],
                y=[spp["atm_pos"][i, 1], spp["atm_pos"][j, 1]],
                z=[spp["atm_pos"][i, 2], spp["atm_pos"][j, 2]],
                mode="lines",
                line=dict(color="black", width=8),
                showlegend=False,
            )
        )

    # Add spheres for each atom (after bonds to ensure Z-order)
    for i, element in enumerate(spp["atm_elmnt"]):
        # Get atomic position and radius
        x_atm = spp["atm_pos"][i, 0]
        y_atm = spp["atm_pos"][i, 1]
        z_atm = spp["atm_pos"][i, 2]
        radius = spp["atm_rad"][i]

        # Get the color for the current element
        color = atom_color_map.get(
            element, "orange"
        )  # Default to green if element not found

        # Add a sphere for the atom
        fig.add_trace(
            go.Scatter3d(
                x=[x_atm],
                y=[y_atm],
                z=[z_atm],
                mode="markers",
                marker=dict(
                    size=radius * 10,  # Scale radius for better visualization
                    color=color,
                ),
                name=None,
                showlegend=False,  # Don't clutter the legend with atom labels
            )
        )

    # Add labels and titles
    fig.update_layout(
        scene=dict(
            xaxis_title="X Position", yaxis_title="Y Position", zaxis_title="Z Position"
        ),
        width=800,
        height=800,
    )

    # Remove grid lines
    fig.update_scenes(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False)

    # Show the plot
    fig.show()

    if mode == "static":
        fig.write_image(os.path.join(dir_plot, plot_name + ".png"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".pdf"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".svg"))
    if mode == "dynamic":
        fig.write_html(os.path.join(dir_plot, plot_name + ".html"))
