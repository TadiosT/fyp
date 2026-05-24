"""Shared Plotly factory for floor-plan + scatter visualisations.

Used by:
- pages/1_Huxley_Labs.py  → workstation status dots
- pages/2_Lecture_Hall.py → live person detections + zone overlay
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
from PIL import Image


def build_floor_map(
    *,
    image: Image.Image,
    points_df: pd.DataFrame,
    color_col: str,
    color_map: dict[str, str],
    category_order: list[str] | None = None,
    hover_template: str | None = None,
    custom_data_cols: list[str] | None = None,
    native_size: tuple[int, int] | None = None,
    marker: dict | None = None,
    height: int = 650,
    extra_shapes: Iterable[dict] | None = None,
    legend_title: str | None = None,
):
    """Return a Plotly figure with `image` as background and `points_df` overlaid.

    Pixel coords are expected in columns `x_px`, `y_px`. Y-axis is inverted so
    image-space pixels (origin top-left) map correctly; `scaleanchor='x'` keeps
    points pinned to image features at any container width.
    """
    w, h = native_size or image.size

    fig = px.scatter(
        points_df,
        x="x_px",
        y="y_px",
        color=color_col,
        color_discrete_map=color_map,
        category_orders={color_col: category_order} if category_order else None,
        custom_data=custom_data_cols,
    )

    marker_cfg = {"size": 12, "line": {"width": 1, "color": "black"}}
    if marker:
        marker_cfg.update(marker)
    trace_update = {"marker": marker_cfg}
    if hover_template:
        trace_update["hovertemplate"] = hover_template
    fig.update_traces(**trace_update)

    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            layer="below",
        )
    )

    if extra_shapes:
        fig.update_layout(shapes=list(extra_shapes))

    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        legend=dict(
            title=legend_title or "",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
