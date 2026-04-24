import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, cKDTree
from rdkit import Chem
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes
from rdkit.Chem import rdDetermineBonds

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

import os
import numpy as np
import plotly.graph_objects as go

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def build_isosurface_fields(points, charges, areas=None, grid_size=40, padding=1.2, sigma=None):
    points = np.asarray(points, dtype=float)
    charges = np.asarray(charges, dtype=float)

    if areas is None:
        weights = np.ones(len(points), dtype=float)
    else:
        areas = np.asarray(areas, dtype=float)
        weights = areas / np.mean(areas)
        weights = np.clip(weights, 0.5, 2.0)

    if sigma is None:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        nn_dist = np.median(dists[:, 1])
        sigma = max(1.5 * nn_dist, 0.25)

    pmin = points.min(axis=0) - padding
    pmax = points.max(axis=0) + padding

    xs = np.linspace(pmin[0], pmax[0], grid_size)
    ys = np.linspace(pmin[1], pmax[1], grid_size)
    zs = np.linspace(pmin[2], pmax[2], grid_size)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    density = np.zeros_like(X, dtype=float)
    charge_num = np.zeros_like(X, dtype=float)

    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for p, q, w in zip(points, charges, weights):
        r2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2
        g = w * np.exp(-r2 * inv_2sigma2)

        density += g
        charge_num += q * g

    charge_field = np.divide(
        charge_num,
        density,
        out=np.zeros_like(charge_num),
        where=density > 1e-12,
    )

    return xs, ys, zs, density, charge_field, sigma


def extract_colored_isosurface(xs, ys, zs, density, charge_field, level):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    verts, faces, _, _ = marching_cubes(
        density,
        level=level,
        spacing=(dx, dy, dz),
    )

    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]

    charge_interp = RegularGridInterpolator(
        (xs, ys, zs),
        charge_field,
        bounds_error=False,
        fill_value=0.0,
    )

    vertex_charge = charge_interp(verts)

    return verts, faces, vertex_charge


def keep_largest_mesh_component(vertices, faces, vertex_values=None):
    n_vertices = len(vertices)
    n_faces = len(faces)

    if n_faces == 0:
        return vertices, faces, vertex_values

    vertex_to_faces = [[] for _ in range(n_vertices)]

    for fi, face in enumerate(faces):
        for v in face:
            vertex_to_faces[v].append(fi)

    visited = np.zeros(n_faces, dtype=bool)
    components = []

    for start_face in range(n_faces):
        if visited[start_face]:
            continue

        stack = [start_face]
        visited[start_face] = True
        comp_faces = []

        while stack:
            fi = stack.pop()
            comp_faces.append(fi)

            for v in faces[fi]:
                for neighbor_face in vertex_to_faces[v]:
                    if not visited[neighbor_face]:
                        visited[neighbor_face] = True
                        stack.append(neighbor_face)

        components.append(comp_faces)

    largest_faces_idx = max(components, key=len)
    largest_faces = faces[largest_faces_idx]

    used_vertices = np.unique(largest_faces.ravel())

    new_index = -np.ones(n_vertices, dtype=int)
    new_index[used_vertices] = np.arange(len(used_vertices))

    new_vertices = vertices[used_vertices]
    new_faces = new_index[largest_faces]

    if vertex_values is None:
        new_vertex_values = None
    else:
        new_vertex_values = vertex_values[used_vertices]

    return new_vertices, new_faces, new_vertex_values


def _combine_meshes(meshes):
    vertices = []
    faces = []
    offset = 0

    for item in meshes:
        if item is None:
            continue

        v, f = item
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)

    if not vertices:
        return None, None

    return np.vstack(vertices), np.vstack(faces)


def _sphere_mesh(center, radius, n_lat=16, n_lon=32):
    center = np.asarray(center, dtype=float)

    n_lat = max(int(n_lat), 6)
    n_lon = max(int(n_lon), 12)

    phi = np.linspace(0, np.pi, n_lat + 1)
    theta = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    vertices = []

    for p in phi:
        sp = np.sin(p)
        cp = np.cos(p)

        for t in theta:
            vertices.append(
                [
                    center[0] + radius * sp * np.cos(t),
                    center[1] + radius * sp * np.sin(t),
                    center[2] + radius * cp,
                ]
            )

    faces = []

    for a in range(n_lat):
        for b in range(n_lon):
            p0 = a * n_lon + b
            p1 = a * n_lon + (b + 1) % n_lon
            p2 = (a + 1) * n_lon + b
            p3 = (a + 1) * n_lon + (b + 1) % n_lon

            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])

    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _cylinder_mesh(p0, p1, radius, n=20):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    n = max(int(n), 8)

    axis = p1 - p0
    length = np.linalg.norm(axis)

    if length < 1e-12:
        return None

    axis = axis / length

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    c0 = []
    c1 = []

    for a in angles:
        radial = radius * (np.cos(a) * u + np.sin(a) * v)
        c0.append(p0 + radial)
        c1.append(p1 + radial)

    vertices = np.vstack([c0, c1, p0[None, :], p1[None, :]])

    start_center = 2 * n
    end_center = 2 * n + 1

    faces = []

    for i in range(n):
        j = (i + 1) % n

        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

        faces.append([start_center, j, i])
        faces.append([end_center, n + i, n + j])

    return vertices, np.asarray(faces, dtype=int)


def _bond_order_count(bond):
    order = bond.GetBondTypeAsDouble()

    if order >= 2.75:
        return 3

    if order >= 1.4:
        return 2

    return 1


def _bond_offset_direction(p0, p1):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    axis = p1 - p0
    length = np.linalg.norm(axis)

    if length < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    axis = axis / length

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    direction = np.cross(axis, ref)
    direction = direction / np.linalg.norm(direction)

    return direction


def _bond_meshes_by_order(p0, p1, order, radius, spacing, n=20):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    direction = _bond_offset_direction(p0, p1)

    if order == 1:
        offsets = [0.0]
    elif order == 2:
        offsets = [-0.5 * spacing, 0.5 * spacing]
    else:
        offsets = [-spacing, 0.0, spacing]

    meshes = []

    for offset in offsets:
        shift = offset * direction
        mesh = _cylinder_mesh(p0 + shift, p1 + shift, radius, n=n)

        if mesh is not None:
            meshes.append(mesh)

    return meshes


def _add_mesh3d(fig, vertices, faces, color, name, visible=True, opacity=1.0):
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            showlegend=False,
            visible=visible,
            flatshading=False,
            lighting=dict(
                ambient=0.72,
                diffuse=0.75,
                specular=0.2,
                roughness=0.55,
                fresnel=0.08,
            ),
        )
    )

    return len(fig.data) - 1


def _fixed_scene_ranges(point_groups, padding=0.8):
    valid_groups = []

    for points in point_groups:
        if points is None:
            continue

        points = np.asarray(points, dtype=float)

        if points.size == 0:
            continue

        valid_groups.append(points.reshape(-1, 3))

    if not valid_groups:
        return [[-1, 1], [-1, 1], [-1, 1]]

    points = np.vstack(valid_groups)

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)

    center = 0.5 * (pmin + pmax)
    half_width = 0.5 * np.max(pmax - pmin) + padding

    if half_width < 1e-6:
        half_width = 1.0

    return [
        [center[0] - half_width, center[0] + half_width],
        [center[1] - half_width, center[1] + half_width],
        [center[2] - half_width, center[2] + half_width],
    ]


def _add_persistent_colorbar(fig, seg_pos, seg_charge, colorscale, colorbar_style):
    p = np.asarray(seg_pos[0], dtype=float)

    fig.add_trace(
        go.Scatter3d(
            x=[p[0], p[0]],
            y=[p[1], p[1]],
            z=[p[2], p[2]],
            mode="markers",
            marker=dict(
                size=0.01,
                color=[float(seg_charge.min()), float(seg_charge.max())],
                colorscale=colorscale,
                cmin=float(seg_charge.min()),
                cmax=float(seg_charge.max()),
                opacity=0.0,
                showscale=True,
                colorbar=colorbar_style,
            ),
            name="Charge scale",
            showlegend=False,
            hoverinfo="skip",
            visible=True,
        )
    )

    return len(fig.data) - 1


def _add_visibility_toggles(
    fig,
    surface_trace_ids,
    structure_trace_ids,
    show_surface=True,
    show_structure=True,
):
    menu_style = dict(
        type="buttons",
        direction="right",
        y=0.985,
        yanchor="top",
        bgcolor="rgba(0,0,0,0.92)",
        bordercolor="rgba(255,255,255,0.55)",
        borderwidth=1,
        font=dict(color="white", size=13),
        pad=dict(l=8, r=8, t=5, b=5),
        showactive=False,
    )

    surface_args = [{"visible": False}, surface_trace_ids]
    surface_args2 = [{"visible": True}, surface_trace_ids]

    if not show_surface:
        surface_args = [{"visible": True}, surface_trace_ids]
        surface_args2 = [{"visible": False}, surface_trace_ids]

    structure_args = [{"visible": False}, structure_trace_ids]
    structure_args2 = [{"visible": True}, structure_trace_ids]

    if not show_structure:
        structure_args = [{"visible": True}, structure_trace_ids]
        structure_args2 = [{"visible": False}, structure_trace_ids]

    fig.update_layout(
        updatemenus=[
            dict(
                menu_style,
                x=0.015,
                xanchor="left",
                active=-1,
                buttons=[
                    dict(
                        label="Toggle surface",
                        method="restyle",
                        args=surface_args,
                        args2=surface_args2,
                    ),
                ],
            ),
            dict(
                menu_style,
                x=0.205,
                xanchor="left",
                active=-1,
                buttons=[
                    dict(
                        label="Toggle structure",
                        method="restyle",
                        args=structure_args,
                        args2=structure_args2,
                    ),
                ],
            ),
        ]
    )

def _fullscreen_html(fig, output_path):
    config = {
        "responsive": True,
        "displaylogo": False,
    }

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config=config,
        default_width="100%",
        default_height="100%",
        div_id="cosmo-surface-plot",
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cosmo Surface</title>
<style>
html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
}}

#cosmo-surface-plot {{
    width: 100vw !important;
    height: 100vh !important;
    background: #000 !important;
}}

.js-plotly-plot,
.plot-container,
.svg-container,
.main-svg {{
    background: #000 !important;
}}

.modebar {{
    background: rgba(18, 18, 18, 0.65) !important;
    border-radius: 10px;
}}

.modebar-btn path {{
    fill: rgba(255, 255, 255, 0.82) !important;
}}

.updatemenu-item-rect,
.updatemenu-header,
.updatemenu-item-rect:hover,
.updatemenu-header:hover,
.updatemenu-item:hover .updatemenu-item-rect,
.updatemenu-button:hover .updatemenu-item-rect {{
    fill: rgba(0, 0, 0, 0.92) !important;
    stroke: rgba(255, 255, 255, 0.55) !important;
}}

.updatemenu-item-text,
.updatemenu-header text,
.updatemenu-item:hover .updatemenu-item-text,
.updatemenu-button:hover .updatemenu-item-text {{
    fill: white !important;
}}
</style>
</head>
<body>
{plot_html}

<script>
function forceDarkToggleButtons() {{
    const plot = document.getElementById("cosmo-surface-plot");

    if (!plot) {{
        return;
    }}

    function applyStyles() {{
        const rects = plot.querySelectorAll(
            ".updatemenu-item-rect, .updatemenu-header"
        );

        rects.forEach((rect) => {{
            rect.style.fill = "rgba(0, 0, 0, 0.92)";
            rect.style.stroke = "rgba(255, 255, 255, 0.55)";
        }});

        const texts = plot.querySelectorAll(
            ".updatemenu-item-text, .updatemenu-header text"
        );

        texts.forEach((text) => {{
            text.style.fill = "white";
        }});
    }}

    applyStyles();

    plot.addEventListener("mouseover", applyStyles);
    plot.addEventListener("mousemove", applyStyles);
    plot.addEventListener("mouseout", applyStyles);
    plot.addEventListener("click", applyStyles);

    const observer = new MutationObserver(applyStyles);
    observer.observe(plot, {{
        attributes: true,
        childList: true,
        subtree: true,
    }});
}}

window.addEventListener("load", forceDarkToggleButtons);
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

def plot_3D_segment_location(
    filepath,
    mode="dynamic",
    dir_plot=".",
    plot_name="cosmo_surface",
    surface_mode="isosurface",
    iso_grid_size=100,
    iso_padding=1.0,
    iso_sigma=None,
    iso_level_fraction=0.28,
    keep_largest_surface_component=True,
    surface_opacity=None,
    show_surface=True,
    show_structure=True,
    atom_radius_scale=0.25,
    bond_radius=0.045,
    bond_spacing=None,
    atom_mesh_resolution=16,
    bond_mesh_resolution=20,
    atom_opacity=1.0,
    bond_opacity=1.0,
):
    spp = SigmaProfileParser(filepath)

    mol = Chem.MolFromXYZBlock(spp.save_to_xyz())
    if mol is None:
        raise ValueError("Unable to load molecule from XYZ file.")

    charge = int(spp["seg_charge"].sum())
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    seg_pos = spp["seg_pos"]
    seg_charge = spp["seg_charge"]
    seg_area = spp["seg_area"]

    surface_colorscale = [
        [0.00, "blue"],
        [0.25, "cyan"],
        [0.50, "lime"],
        [0.70, "yellow"],
        [1.00, "red"],
    ]

    colorbar_style = dict(
        title=dict(text="Charge", font=dict(color="white")),
        tickfont=dict(color="white"),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.35)",
    )

    if surface_opacity is None:
        surface_opacity = 0.25 if surface_mode == "isosurface" else 0.45

    if bond_spacing is None:
        bond_spacing = bond_radius * 3.0

    fig = go.Figure()

    surface_trace_ids = []
    structure_trace_ids = []

    scene_points = [
        seg_pos,
        spp["atm_pos"],
    ]

    _add_persistent_colorbar(
        fig,
        seg_pos,
        seg_charge,
        surface_colorscale,
        colorbar_style,
    )

    if surface_mode == "points":
        fig.add_trace(
            go.Scatter3d(
                x=seg_pos[:, 0],
                y=seg_pos[:, 1],
                z=seg_pos[:, 2],
                mode="markers",
                marker=dict(
                    size=seg_area * 50,
                    color=seg_charge,
                    colorscale=surface_colorscale,
                    cmin=seg_charge.min(),
                    cmax=seg_charge.max(),
                    opacity=surface_opacity,
                    showscale=False,
                ),
                name="Surface",
                showlegend=False,
                visible=show_surface,
            )
        )

        surface_trace_ids.append(len(fig.data) - 1)

    elif surface_mode == "isosurface":
        xs, ys, zs, density, charge_field, _ = build_isosurface_fields(
            seg_pos,
            seg_charge,
            areas=seg_area,
            grid_size=iso_grid_size,
            padding=iso_padding,
            sigma=iso_sigma,
        )

        iso_level = iso_level_fraction * density.max()

        verts, faces, vertex_charge = extract_colored_isosurface(
            xs,
            ys,
            zs,
            density,
            charge_field,
            iso_level,
        )

        if keep_largest_surface_component:
            verts, faces, vertex_charge = keep_largest_mesh_component(
                verts,
                faces,
                vertex_charge,
            )

        scene_points.append(verts)

        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=vertex_charge,
                intensitymode="vertex",
                colorscale=surface_colorscale,
                cmin=seg_charge.min(),
                cmax=seg_charge.max(),
                opacity=surface_opacity,
                showscale=False,
                name="Surface",
                showlegend=False,
                visible=show_surface,
                flatshading=False,
                lighting=dict(
                    ambient=0.75,
                    diffuse=0.45,
                    specular=0.05,
                    roughness=0.9,
                    fresnel=0.0,
                ),
            )
        )

        surface_trace_ids.append(len(fig.data) - 1)

    else:
        raise ValueError("surface_mode must be either 'points' or 'isosurface'.")

    atoms_available = sorted(set(spp["atm_elmnt"]))
    atom_color_map = get_atom_color_map(atoms_available)

    bond_meshes = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        order = _bond_order_count(bond)

        bond_meshes.extend(
            _bond_meshes_by_order(
                spp["atm_pos"][i],
                spp["atm_pos"][j],
                order=order,
                radius=bond_radius,
                spacing=bond_spacing,
                n=bond_mesh_resolution,
            )
        )

    bond_vertices, bond_faces = _combine_meshes(bond_meshes)

    if bond_vertices is not None:
        scene_points.append(bond_vertices)

        trace_id = _add_mesh3d(
            fig,
            bond_vertices,
            bond_faces,
            color="rgb(235,235,235)",
            name="Bonds",
            visible=show_structure,
            opacity=bond_opacity,
        )

        structure_trace_ids.append(trace_id)

    atom_meshes_by_color = {}

    for i, element in enumerate(spp["atm_elmnt"]):
        center = spp["atm_pos"][i]
        radius = spp["atm_rad"][i] * atom_radius_scale
        color = atom_color_map.get(element, "orange")

        key = (element, color)

        atom_meshes_by_color.setdefault(key, []).append(
            _sphere_mesh(
                center,
                radius,
                n_lat=atom_mesh_resolution,
                n_lon=atom_mesh_resolution * 2,
            )
        )

    for (element, color), meshes in atom_meshes_by_color.items():
        atom_vertices, atom_faces = _combine_meshes(meshes)

        scene_points.append(atom_vertices)

        trace_id = _add_mesh3d(
            fig,
            atom_vertices,
            atom_faces,
            color=color,
            name=f"{element} atoms",
            visible=show_structure,
            opacity=atom_opacity,
        )

        structure_trace_ids.append(trace_id)

    _add_visibility_toggles(
        fig,
        surface_trace_ids,
        structure_trace_ids,
        show_surface=show_surface,
        show_structure=show_structure,
    )

    x_range, y_range, z_range = _fixed_scene_ranges(
        scene_points,
        padding=max(iso_padding, 1.0),
    )

    hidden_axis = dict(
        visible=False,
        showgrid=False,
        showbackground=False,
        showline=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        title="",
        autorange=False,
    )

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="fixed-scene",
        scene=dict(
            bgcolor="black",
            xaxis=dict(hidden_axis, range=x_range),
            yaxis=dict(hidden_axis, range=y_range),
            zaxis=dict(hidden_axis, range=z_range),
            aspectmode="cube",
            camera=dict(
                projection=dict(type="orthographic"),
            ),
        ),
    )

    config = {
        "responsive": True,
        "displaylogo": False,
    }

    fig.show(config=config)

    if mode == "static":
        fig.write_image(os.path.join(dir_plot, plot_name + ".png"), scale=2)
        fig.write_image(os.path.join(dir_plot, plot_name + ".pdf"))
        fig.write_image(os.path.join(dir_plot, plot_name + ".svg"))

    if mode == "dynamic":
        output_path = os.path.join(dir_plot, plot_name + ".html")
        _fullscreen_html(fig, output_path)