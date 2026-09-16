import pandas as pd
import pytest
from PIL import Image

from campus_occupancy.dashboard.floor_map import build_floor_map


@pytest.fixture
def image():
    return Image.new("RGB", (100, 80), "white")


@pytest.fixture
def points():
    return pd.DataFrame({"x_px": [10, 50, 90], "y_px": [10, 40, 70],
                         "state": ["In Use", "Empty", "In Use"], "pc_id": ["a", "b", "c"]})


COLOURS = {"In Use": "#32CD32", "Empty": "#FF0000"}


def test_background_image_and_axes(image, points):
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS)
    img = fig.layout.images[0]
    assert (img.sizex, img.sizey, img.x, img.y, img.layer) == (100, 80, 0, 0, "below")
    assert tuple(fig.layout.yaxis.range) == (80, 0) and fig.layout.yaxis.scaleanchor == "x"
    assert tuple(fig.layout.xaxis.range) == (0, 100)
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False


def test_native_size_override_and_height(image, points):
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS,
                          native_size=(400, 320), height=777)
    assert (fig.layout.images[0].sizex, fig.layout.images[0].sizey) == (400, 320)
    assert tuple(fig.layout.yaxis.range) == (320, 0) and fig.layout.height == 777


def test_one_trace_per_category_in_order(image, points):
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS,
                          category_order=["Empty", "In Use"])
    assert [t.name for t in fig.data] == ["Empty", "In Use"]
    assert fig.data[1].marker.color == "#32CD32"


def test_hover_template_and_customdata(image, points):
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS,
                          custom_data_cols=["pc_id"], hover_template="<b>%{customdata[0]}</b><extra></extra>")
    assert all(t.hovertemplate == "<b>%{customdata[0]}</b><extra></extra>" for t in fig.data)
    assert fig.data[0].customdata is not None


def test_marker_overrides_merge_with_defaults(image, points):
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS,
                          marker={"size": 5, "opacity": 0.5})
    m = fig.data[0].marker
    assert m.size == 5 and m.opacity == 0.5 and m.line.width == 1


def test_extra_shapes_and_legend_title(image, points):
    shape = dict(type="rect", x0=0, x1=100, y0=0, y1=40, fillcolor="rgba(255,215,0,0.2)")
    fig = build_floor_map(image=image, points_df=points, color_col="state", color_map=COLOURS,
                          extra_shapes=[shape], legend_title="Seat state")
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y1 == 40
    assert fig.layout.legend.title.text == "Seat state"


def test_empty_points(image):
    empty = pd.DataFrame(columns=["x_px", "y_px", "state"])
    fig = build_floor_map(image=image, points_df=empty, color_col="state", color_map=COLOURS)
    assert len(fig.layout.images) == 1
