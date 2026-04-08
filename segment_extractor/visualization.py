"""
Matplotlib/GeoPandas plotting helpers.
"""
from __future__ import annotations

import os
from typing import Optional

import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless/background jobs

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PathCollection, PolyCollection



mpl.rcParams.update(
    {   "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

def apply_white_halo_to_axes(
    ax,
    *,
    halo_color="white",
    line_factor=1.2,
    min_line_halo=4.0,
    marker_halo=3.5,
    poly_halo=None,
):
    """
    Apply white halo to plotted lines/points/polygons for better presentation.
    """
    for line in ax.lines:
        lw = float(line.get_linewidth() or 1.0)
        halo_w = max(min_line_halo, lw * line_factor)
        line.set_path_effects([pe.Stroke(linewidth=halo_w, foreground=halo_color), pe.Normal()])

    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            lws = coll.get_linewidths()
            base_lw = float(max(lws) if len(lws) else 1.0)
            halo_w = max(min_line_halo, base_lw * line_factor)
            coll.set_path_effects([pe.Stroke(linewidth=halo_w, foreground=halo_color), pe.Normal()])
        elif isinstance(coll, PathCollection):
            coll.set_path_effects([pe.Stroke(linewidth=marker_halo, foreground=halo_color), pe.Normal()])
        elif isinstance(coll, PolyCollection) and poly_halo is not None:
            coll.set_path_effects([pe.Stroke(linewidth=poly_halo, foreground=halo_color), pe.Normal()])


def debug_plot_all_ways(result, projection=None):
    """
    Quick diagnostic plots of Overpass ways in raw/projection space.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True) if projection else None
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_raw, ax_proj = axes

    for w in result.ways:
        coords = [(float(n.lon), float(n.lat)) for n in w.nodes]
        xs, ys = zip(*coords)
        ax_raw.plot(xs, ys, color="gray", alpha=0.3)

    ax_raw.set_aspect("equal")
    ax_raw.set_xlabel("Longitude")
    ax_raw.set_ylabel("Latitude")
    ax_raw.set_title("All Ways (Raw EPSG:4326)")
    ax_raw.grid(True)

    for w in result.ways:
        coords = [(float(n.lon), float(n.lat)) for n in w.nodes]
        xs, ys = zip(*coords)
        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)
        ax_proj.plot(xs, ys, color="gray", alpha=0.3)

    ax_proj.set_aspect("equal")
    if projection:
        ax_proj.set_xlabel(f"X ({projection})")
        ax_proj.set_ylabel(f"Y ({projection})")
        ax_proj.set_title(f"All Ways (Projected: {projection})")
    else:
        ax_proj.set_xlabel("Longitude")
        ax_proj.set_ylabel("Latitude")
        ax_proj.set_title("All Ways (No Reprojection)")
    ax_proj.grid(True)
    plt.tight_layout()
    plt.show()


def debug_plot_gdf(
    gdf,
    title,
    projection,
    out_filename,
    out_dir="outputs/intermediates",
    figsize=(6, 6),
    color="#0F172A",
    alpha=0.95,
    linewidth=1.1,
    markersize=2.8,
    basemap=cx.providers.OpenStreetMap.Mapnik,
    basemap_alpha=0.85,
    show_legend=False,
    legend_label=None,
    legend_loc="upper right",
    legend_frameon=True,
    axis_off=True,
    pad=0.2,
    dpi=450,
    save=True,
    image_ext="svg",
):
    """
    Basemap figure helper used by debug plotting steps.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS.")
    if projection is None:
        raise ValueError("projection is required.")

    gdf_proj = gdf.to_crs(projection)
    gdf_3857 = gdf_proj.to_crs("EPSG:3857")

    legend_text = legend_label or title or "Selection"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#f8fafc")
    gdf_3857.plot(ax=ax, color=color, alpha=alpha, linewidth=linewidth, markersize=markersize, zorder=3)
    cx.add_basemap(ax, source=basemap, crs="EPSG:3857", alpha=basemap_alpha, attribution="© OpenStreetMap contributors")

    try:
        xmin, ymin, xmax, ymax = gdf_3857.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    except Exception:
        pass

    ax.set_title(f"{title}", fontsize=12)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()

    if show_legend:
        handles = []
        geom_types = gdf_3857.geom_type.value_counts()
        dominant = geom_types.index[0] if len(geom_types) else None
        is_pointy = dominant in {"Point", "MultiPoint"}
        is_liney = dominant in {"LineString", "MultiLineString"}
        if is_liney:
            handles.append(mlines.Line2D([], [], color=color, linewidth=linewidth, label=legend_text))
        elif is_pointy:
            handles.append(
                mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=10, label=legend_text)
            )
        else:
            handles.append(mpatches.Patch(color=color, label=legend_text))
        if handles:
            legend = ax.legend(handles=handles, loc=legend_loc, frameon=legend_frameon, fontsize=12)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.9)

    fig.tight_layout()
    if save:
        os.makedirs(out_dir, exist_ok=True)
        ext = image_ext.lstrip(".") or "png"
        fname = f"{out_filename}.{ext}"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return out_path

    return fig, ax


def plot_all_data(
    mainline_gdf,
    ramps_gdf,
    lanes_gdf,
    out_pdf_path,
    *,
    title=None,
    figsize=(5, 5),
    pad=0.2,
    dpi=300,
    basemap_provider="CartoDB.Positron",
    buffer_ratio=0.08,
):
    """
    Plot mainline + ramps + lane-change sites for validation.
    """
    if mainline_gdf.empty:
        raise ValueError("mainline_gdf is empty.")
    if ramps_gdf.empty:
        raise ValueError("ramps_gdf is empty.")
    if lanes_gdf.empty:
        raise ValueError("lane_changes_gdf is empty.")
    if mainline_gdf.crs is None or ramps_gdf.crs is None or lanes_gdf.crs is None:
        raise ValueError("All GeoDataFrames must have a valid .crs set.")

    main_3857 = mainline_gdf.to_crs(epsg=3857)
    ramps_3857 = ramps_gdf.to_crs(epsg=3857)
    lanes_3857 = lanes_gdf.to_crs(epsg=3857)

    mainline_color = "#1d3557"
    entry_color = "#2a9d8f"
    exit_color = "#e76f51"
    lane_color = "#f4a261"

    for col in ("entry_node", "exit_node"):
        if col not in ramps_3857.columns:
            raise ValueError(f"ramps_gdf is missing required column '{col}'.")

    entry_flag = ramps_3857["entry_node"].fillna(False).astype(bool)
    exit_flag = ramps_3857["exit_node"].fillna(False).astype(bool)

    ramps_entry = ramps_3857[entry_flag & ~exit_flag]
    ramps_exit = ramps_3857[~entry_flag & exit_flag]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8fafc")
    main_3857.plot(ax=ax, linewidth=4.2, color=mainline_color, alpha=0.95, zorder=4, label="Mainline path")
    apply_white_halo_to_axes(ax)

    lanes_3857.plot(ax=ax, color=lane_color, markersize=90, marker="D", zorder=6, label="Lane measurements")
    apply_white_halo_to_axes(ax, marker_halo=2)

    if not ramps_entry.empty:
        ramps_entry.plot(ax=ax, color=entry_color, markersize=80, marker="o", zorder=7, label="On-ramps")
        apply_white_halo_to_axes(ax, marker_halo=2)
    if not ramps_exit.empty:
        ramps_exit.plot(ax=ax, color=exit_color, markersize=80, marker="o", zorder=7, label="Off-ramps")
        apply_white_halo_to_axes(ax, marker_halo=2)

    provider = cx.providers.OpenStreetMap.Mapnik if basemap_provider == "CartoDB.Positron" else eval(f"cx.providers.{basemap_provider}")
    cx.add_basemap(ax, source=provider, attribution_size=7, zorder=1, alpha=0.9)

    try:
        xmin, ymin, xmax, ymax = main_3857.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    except Exception:
        pass

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=20, pad=12)

    handles = [
        mlines.Line2D([], [], color=mainline_color, linewidth=4, label="Mainline path"),
        mlines.Line2D([], [], color=lane_color, marker="D", linestyle="None", markersize=10, label="Lane RCS"),
        mlines.Line2D([], [], color=entry_color, marker="o", linestyle="None", markersize=10, label="On-ramps"),
        mlines.Line2D([], [], color=exit_color, marker="o", linestyle="None", markersize=10, label="Off-ramps"),
    ]
    legend = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    fig.savefig(out_pdf_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_pdf_path
