import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rdkit.Chem
import rdkit.Chem.rdDetermineBonds

from opencosmorspy.input_parsers import SigmaProfileParser




def plot_sigmaprofiles(
    filepath_lst,
    plot_name=None,
    plot_label_dct=None,
    xlim=(-0.02, 0.02),
    dir_plot=None,
    single_plot=False
):

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

        if single_plot:
            if dir_plot:
                fig.savefig(os.path.join(dir_plot,  f"{plot_name}_{label}.png"), dpi=300)
            else:
                plt.show()

    if not single_plot:
        if dir_plot:
            fig.savefig(os.path.join(dir_plot,  f"{plot_name}_{label}.png"), dpi=300)
        else:
            plt.show()


def plot_sigmaprofiles_plotly(
    filepath_lst,
    plot_name=None,
    plot_label_dct=None,
    xlim=(-0.02, 0.02),
    dir_plot=None,
    mode="static"
):

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


def plot_extended_sigmaprofiles_plotly(dir_plot, filepath, qc_program, mode="dynamic"):

    plot_name = "extsp_" + os.path.splitext(os.path.basename(filepath))[0]

    df_esp = _generate_extended_sigma_profile(filepath, qc_program)

    df_esp["color"] = df_esp["elmnt_nr"]
    df_esp.loc[df_esp["color"] == 106, "color"] = "H-C"
    df_esp.loc[df_esp["color"] == 108, "color"] = "H-O"
    df_esp.loc[df_esp["color"] == 107, "color"] = "H-N"
    df_esp.loc[df_esp["color"] == 6, "color"] = "C"
    df_esp.loc[df_esp["color"] == 8, "color"] = "O"
    df_esp.loc[df_esp["color"] == 7, "color"] = "N"
    df_esp.loc[df_esp["color"] == 17, "color"] = "Cl"

    # Make small areas visible, as they are clusters
    df_esp.loc[(df_esp["area"] != 0) & (df_esp["area"] < 0.06), "area"] = 0.06

    # Pivot
    df_temp = pd.pivot_table(df_esp, values="sigma_orth", index="sigma", aggfunc=np.sum)
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

    fig = px.scatter(
        df_esp,
        x="sigma",
        y="sigma_orth",
        color="color",
        labels=labels,
        size="area",
        hover_data=["sigma", "sigma_orth", "elmnt_nr"],
        size_max=40,
        opacity=0.7,
        color_discrete_map={
            "H-C": "#316395",
            "H-O": "#AF0000",
            "H-N": "#7600b5",
            "C": "#109618",
            "O": "#565656",
            "N": "#DEBC00",
            "Cl": "#09118C",
        },
        category_orders={"color": ["H-O", "H-N", "H-C", "Cl", "C", "N", "O", "Cl"]},
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


def plot_cosmo_surface(parser, mode="dynamic", dir_plot=".", plot_name="cosmo_surface"):
    """Plot a COSMO surface with atoms and bonds.

    Example usage:
    parser = SigmaProfileParser('path/to/simulation.orcacosmo', qc_program='orca')
    plot_cosmo_surface_with_atoms_and_bonds(parser)"""

    mol = rdkit.Chem.MolFromXYZBlock(parser.save_to_xyz())
    rdkit.Chem.rdDetermineBonds.DetermineBonds(mol, charge=0)
    if mol is None:
        raise ValueError("Unable to load molecule from XYZ file.")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=parser["seg_pos"][:, 0],
                y=parser["seg_pos"][:, 1],
                z=parser["seg_pos"][:, 2],
                mode="markers",
                marker=dict(
                    size=parser["seg_area"]
                    * 50,  # Scale the size for better visualization
                    color=parser["seg_charge"],  # Set the color based on charge
                    colorscale="bluered",  # Color scale for charges
                    colorbar=dict(title="Charge"),
                    opacity=0.8,
                ),
                name=None,
                showlegend=False,
            )
        ]
    )

    element_colors = {
        "H": "green",  # Hydrogen
        "C": "black",  # Carbon
        "N": "blue",  # Nitrogen
        "O": "red",  # Oxygen
        "S": "yellow",  # Sulfur
        "P": "orange",  # Phosphorus
        "F": "light green",  # Fluorine
        "Cl": "green",  # Chlorine
        "Br": "brown",  # Bromine
        "I": "purple",  # Iodine
        "He": "cyan",  # Helium
        "Ne": "cyan",  # Neon
        "Ar": "cyan",  # Argon
        "Li": "dark red",  # Lithium
        "Na": "blue",  # Sodium
        "K": "purple",  # Potassium
        "Ca": "dark green",  # Calcium
        "Fe": "orange",  # Iron
        "Mg": "gray",  # Magnesium
    }

    # Add bonds as lines (RDKit automatically infers bonds from 3D structure)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add bond (line) between the two atoms
        fig.add_trace(
            go.Scatter3d(
                x=[parser["atm_pos"][i, 0], parser["atm_pos"][j, 0]],
                y=[parser["atm_pos"][i, 1], parser["atm_pos"][j, 1]],
                z=[parser["atm_pos"][i, 2], parser["atm_pos"][j, 2]],
                mode="lines",
                line=dict(color="black", width=10),
                showlegend=False,
            )
        )

    # Add spheres for each atom (after bonds to ensure Z-order)
    for i, element in enumerate(parser["atm_elmnt"]):
        # Get atomic position and radius
        x_atm = parser["atm_pos"][i, 0]
        y_atm = parser["atm_pos"][i, 1]
        z_atm = parser["atm_pos"][i, 2]
        radius = parser["atm_rad"][i]

        # Get the color for the current element
        color = element_colors.get(
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


if __name__ == "__main__":
    pass
