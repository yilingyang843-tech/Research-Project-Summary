import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

DEFAULT_FUNC = "quadratic"
DEFAULT_A = 1.0
DEFAULT_B = 0.0
DEFAULT_C = 0.0
DEFAULT_X_RANGE = [-10, 10]

FUNC_LABELS = {
    "linear": "Linear",
    "quadratic": "Quadratic",
    "sine": "Sine",
}


def build_formula(func_type, a, b, c):
    if func_type == "linear":
        return f"y = {a:.2f}·x + {b:.2f}"
    elif func_type == "quadratic":
        return f"y = {a:.2f}·x² + {b:.2f}·x + {c:.2f}"
    else:  # sine
        return f"y = {a:.2f}·sin({b:.2f}·x) + {c:.2f}"


def build_figure(func_type, a, b, c, x_range, show_grid, show_keys, color_theme):
    x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, 400)

    if func_type == "linear":
        ys = a * xs + b
    elif func_type == "quadratic":
        ys = a * xs ** 2 + b * xs + c
    else:  # sine
        ys = a * np.sin(b * xs) + c

    if color_theme == "green":
        line_color = "#2ca02c"
    elif color_theme == "red":
        line_color = "#d62728"
    elif color_theme == "dark":
        line_color = "#1f77b4"
    else:  # blue
        line_color = "#1f77b4"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color=line_color, width=3),
            marker=dict(size=4, opacity=0.25),
            name="Function",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )

    # y 轴范围
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    pad_y = (y_max - y_min) * 0.1 + 1.0
    y0, y1 = y_min - pad_y, y_max + pad_y

    # x 轴 / y 轴线
    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=0,
        y1=0,
        line=dict(color="black", width=1),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=y0,
        y1=y1,
        line=dict(color="black", width=1, dash="dot"),
        layer="below",
    )

    if show_keys:
        idx_max = int(np.argmax(ys))
        idx_min = int(np.argmin(ys))
        x_max_pt, y_max_pt = xs[idx_max], ys[idx_max]
        x_min_pt, y_min_pt = xs[idx_min], ys[idx_min]

        # Max
        fig.add_trace(
            go.Scatter(
                x=[x_max_pt],
                y=[y_max_pt],
                mode="markers+text",
                name="Max",
                marker=dict(color="orange", size=9),
                text=["Max"],
                textposition="top center",
                hovertemplate="Max<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

        # Min
        fig.add_trace(
            go.Scatter(
                x=[x_min_pt],
                y=[y_min_pt],
                mode="markers+text",
                name="Min",
                marker=dict(color="purple", size=9),
                text=["Min"],
                textposition="bottom center",
                hovertemplate="Min<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

        # y 截距
        if func_type == "linear":
            y_intercept = b
        elif func_type == "quadratic":
            y_intercept = c
        else:
            y_intercept = a * np.sin(b * 0) + c

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y_intercept],
                mode="markers+text",
                name="y-intercept",
                marker=dict(color="black", size=9),
                text=["y-intercept"],
                textposition="bottom center",
                hovertemplate="y-intercept<br>x=0.00<br>y=%{y:.2f}<extra></extra>",
            )
        )

    # 主题 & 网格
    if color_theme == "dark":
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="#f2f2f2"),
        )
        grid_color = "#444444"
    else:
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#222222"),
        )
        grid_color = "#dddddd"

    fig.update_xaxes(
        title="x",
        range=[x_min, x_max],
        showgrid=show_grid,
        gridcolor=grid_color,
        zeroline=False,
        fixedrange=False,  # ✅ 允许缩放
    )
    fig.update_yaxes(
        title="y",
        range=[y0, y1],
        showgrid=show_grid,
        gridcolor=grid_color,
        zeroline=False,
        fixedrange=False,  # ✅ 允许缩放
    )

    func_label = FUNC_LABELS[func_type]
    formula = build_formula(func_type, a, b, c)

    fig.update_layout(
        title=f"Current Function: {func_label}, {formula}",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="closest",
        dragmode="zoom",  # ✅ 默认拖动为 Zoom
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )

    return fig, formula


def build_analysis(func_type, a, b, c):
    items = []
    formula = build_formula(func_type, a, b, c)

    if func_type == "linear":
        first_deriv = f"f'(x) = {a:.2f}"
        second_deriv = "f''(x) = 0.00 (constant)"

        if a > 0:
            mono = "The function is strictly increasing on the whole real line because a > 0."
        elif a < 0:
            mono = "The function is strictly decreasing on the whole real line because a < 0."
        else:
            mono = "The function is constant (a = 0)."

        concavity = "Linear functions are both concave and convex (second derivative is zero)."

        if a != 0:
            x_zero = -b / a
            zeros = f"Zero (x-intercept) at x = -b/a = {x_zero:.2f}."
        else:
            zeros = "No unique zero when a = 0 (either no solution or infinitely many)."

        if abs(b) < 1e-6:
            parity = "Parity: the function is odd (f(-x) = -f(x)) because it passes through the origin (b ≈ 0)."
        else:
            parity = "Parity: the function is neither even nor odd in general when b ≠ 0."

        items.extend(
            [
                html.Li(f"Type: Linear function, {formula}."),
                html.Li(first_deriv),
                html.Li(second_deriv),
                html.Li(mono),
                html.Li(concavity),
                html.Li(zeros),
                html.Li(parity),
            ]
        )

    elif func_type == "quadratic":
        first_deriv = f"f'(x) = {2 * a:.2f}·x + {b:.2f}."
        second_deriv = f"f''(x) = {2 * a:.2f} (constant)."

        if abs(a) > 1e-8:
            x_vertex = -b / (2 * a)
            y_vertex = a * x_vertex ** 2 + b * x_vertex + c
            vertex = f"Extrema: vertex at x = -b/(2a) = {x_vertex:.2f}, y = {y_vertex:.2f}."
            if a > 0:
                vertex += " This is a global minimum (parabola opens upward, a > 0)."
            else:
                vertex += " This is a global maximum (parabola opens downward, a < 0)."
        else:
            vertex = "When a = 0 the function degenerates to a linear function (no quadratic vertex)."

        if a > 0:
            mono = "Monotonicity: decreasing on (-∞, x_vertex), increasing on (x_vertex, +∞)."
        elif a < 0:
            mono = "Monotonicity: increasing on (-∞, x_vertex), decreasing on (x_vertex, +∞)."
        else:
            mono = "Monotonicity: same as a linear function when a = 0."

        if a > 0:
            concavity = "Concavity: f(x) is concave up (opens upward) because a > 0."
        elif a < 0:
            concavity = "Concavity: f(x) is concave down (opens downward) because a < 0."
        else:
            concavity = "Concavity: second derivative is zero when a = 0 (no curvature)."

        if abs(a) > 1e-8:
            D = b ** 2 - 4 * a * c
            if D > 0:
                r1 = (-b + np.sqrt(D)) / (2 * a)
                r2 = (-b - np.sqrt(D)) / (2 * a)
                zeros = f"Zeros: two real roots at x ≈ {r1:.2f} and x ≈ {r2:.2f} (discriminant > 0)."
            elif D == 0:
                r = -b / (2 * a)
                zeros = f"Zeros: one repeated real root at x ≈ {r:.2f} (discriminant = 0)."
            else:
                zeros = "Zeros: no real roots (discriminant < 0)."
        else:
            zeros = "Zeros: same as the corresponding linear function (a = 0)."

        if abs(b) < 1e-6:
            parity = "Parity: the function is even (f(-x) = f(x)) when the linear term b·x is approximately zero."
        else:
            parity = "Parity: in general a·x² + b·x + c is neither even nor odd when b ≠ 0."

        items.extend(
            [
                html.Li(f"Type: Quadratic function, {formula}."),
                html.Li(first_deriv),
                html.Li(second_deriv),
                html.Li(vertex),
                html.Li(mono),
                html.Li(concavity),
                html.Li(zeros),
                html.Li(parity),
            ]
        )

    else:  # sine
        first_deriv = f"f'(x) = {a * b:.2f}·cos({b:.2f}·x)."
        second_deriv = f"f''(x) = {-a * b ** 2:.2f}·sin({b:.2f}·x)."

        amp = abs(a)
        if b != 0:
            period = 2 * np.pi / abs(b)
            period_text = f"Fundamental period: T = 2π/|b| ≈ {period:.2f}."
        else:
            period_text = "When b = 0, the function becomes constant in x (no oscillation)."

        mono = (
            "The function oscillates between its maximum and minimum; "
            "it is not globally increasing or decreasing."
        )
        concavity = (
            "Concavity alternates: f''(x) changes sign periodically, producing alternating "
            "intervals of concave up and concave down."
        )

        if abs(c) < 1e-6:
            parity = "Parity: y = a·sin(bx) is an odd function when there is no vertical shift (c ≈ 0)."
        else:
            parity = (
                "Parity: adding a vertical shift c makes the function neither even nor odd in general."
            )

        items.extend(
            [
                html.Li(f"Type: Sine function, {formula}."),
                html.Li(f"Amplitude: |a| = {amp:.2f}."),
                html.Li(period_text),
                html.Li(first_deriv),
                html.Li(second_deriv),
                html.Li(mono),
                html.Li(concavity),
                html.Li(parity),
            ]
        )

    return html.Ul(items, style={"marginTop": 0})


# ==== Dash App ====

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1(
            "Interactive Function Explorer",
            style={"textAlign": "center", "marginTop": "20px", "marginBottom": "10px"},
        ),

        # 选择函数类型
        html.Div(
            [
                html.Label("Select Function Type:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="func-type",
                    options=[
                        {"label": "Linear (y = a·x + b)", "value": "linear"},
                        {"label": "Quadratic (y = a·x² + b·x + c)", "value": "quadratic"},
                        {"label": "Sine (y = a·sin(bx) + c)", "value": "sine"},
                    ],
                    value=DEFAULT_FUNC,
                    clearable=False,
                    style={"width": "320px"},
                ),
            ],
            style={"marginBottom": "15px"},
        ),

        # 参数滑块区域
        html.Div(
            [
                html.H3("Parameters", style={"marginTop": 0}),

                html.Label("Parameter a (controls slope / opening / amplitude):"),
                dcc.Slider(
                    id="slider-a",
                    min=-10,
                    max=10,
                    step=0.5,
                    value=DEFAULT_A,
                    marks={-10: "-10", -5: "-5", 0: "0", 5: "5", 10: "10"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),

                html.Label("Parameter b (controls shift / linear term / frequency):"),
                dcc.Slider(
                    id="slider-b",
                    min=-10,
                    max=10,
                    step=0.5,
                    value=DEFAULT_B,
                    marks={-10: "-10", -5: "-5", 0: "0", 5: "5", 10: "10"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),

                html.Label("Parameter c (vertical shift / constant term):"),
                dcc.Slider(
                    id="slider-c",
                    min=-10,
                    max=10,
                    step=0.5,
                    value=DEFAULT_C,
                    marks={-10: "-10", -5: "-5", 0: "0", 5: "5", 10: "10"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),

                html.Label("X Range (controls visible window on x-axis):"),
                dcc.RangeSlider(
                    id="x-range",
                    min=-30,
                    max=30,
                    step=1,
                    value=DEFAULT_X_RANGE,
                    marks={-30: "-30", -20: "-20", -10: "-10", 0: "0", 10: "10", 20: "20", 30: "30"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={
                "padding": "15px",
                "border": "1px solid #dddddd",
                "borderRadius": "10px",
                "backgroundColor": "#fafafa",
                "marginBottom": "15px",
            },
        ),

        # 显示选项 + 颜色 + Reset
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Display Options:", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="show-grid",
                            options=[{"label": " Show grid", "value": "grid"}],
                            value=["grid"],
                            style={"marginBottom": "4px"},
                        ),
                        dcc.Checklist(
                            id="show-keys",
                            options=[{"label": " Show key points (max/min/intercepts)", "value": "keys"}],
                            value=["keys"],
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Color Theme:", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="color-theme",
                            options=[
                                {"label": " Blue", "value": "blue"},
                                {"label": " Green", "value": "green"},
                                {"label": " Red", "value": "red"},
                                {"label": " Dark", "value": "dark"},
                            ],
                            value="blue",
                        ),
                    ],
                    style={"flex": "1", "paddingLeft": "30px"},
                ),
                html.Div(
                    [
                        html.Button("Reset", id="reset-button", n_clicks=0),
                    ],
                    style={"flex": "0 0 auto", "textAlign": "right"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-start",
                "marginBottom": "10px",
            },
        ),

        dcc.Graph(
            id="graph",
            style={"height": "450px"},
            config={
                "displaylogo": False,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                "scrollZoom": True,  # ✅ 鼠标滚轮缩放
            },
        ),

        # Selected points + Function Analysis
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Selected points"),
                        html.Div(
                            "Selected points: 0. No points selected. Use Box Select or Lasso Select on the graph toolbar to highlight points.",
                            id="selected-points",
                        ),
                    ],
                    style={
                        "flex": 1,
                        "padding": "10px",
                        "border": "1px solid #dddddd",
                        "borderRadius": "8px",
                        "marginRight": "10px",
                        "backgroundColor": "#fafafa",
                    },
                ),
                html.Div(
                    [
                        html.H3("Function Analysis"),
                        html.Div(id="analysis"),
                    ],
                    style={
                        "flex": 1,
                        "padding": "10px",
                        "border": "1px solid #dddddd",
                        "borderRadius": "8px",
                        "backgroundColor": "#fafafa",
                    },
                ),
            ],
            style={"display": "flex", "marginTop": "15px", "marginBottom": "15px"},
        ),

        # 历史记录
        html.Div(
            [
                html.H3("Function History"),
                html.Div(
                    "Recent functions you explored will appear here.",
                    id="history",
                ),
            ],
            style={
                "border": "1px solid #dddddd",
                "borderRadius": "8px",
                "padding": "10px",
                "backgroundColor": "#fafafa",
                "marginBottom": "25px",
            },
        ),

        dcc.Store(id="history-store", data=[]),
    ],
)


# ==== 回调：更新图像 + 分析 ====

@app.callback(
    Output("graph", "figure"),
    Output("analysis", "children"),
    Input("func-type", "value"),
    Input("slider-a", "value"),
    Input("slider-b", "value"),
    Input("slider-c", "value"),
    Input("x-range", "value"),
    Input("show-grid", "value"),
    Input("show-keys", "value"),
    Input("color-theme", "value"),
)
def update_graph_and_analysis(func_type, a, b, c, x_range, show_grid_val, show_keys_val, color_theme):
    show_grid = "grid" in (show_grid_val or [])
    show_keys = "keys" in (show_keys_val or [])

    fig, _ = build_figure(func_type, a, b, c, x_range, show_grid, show_keys, color_theme)
    analysis = build_analysis(func_type, a, b, c)
    return fig, analysis


# ==== 回调：Reset ====

@app.callback(
    Output("func-type", "value"),
    Output("slider-a", "value"),
    Output("slider-b", "value"),
    Output("slider-c", "value"),
    Output("x-range", "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_controls(n_clicks):
    return DEFAULT_FUNC, DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_X_RANGE


# ==== 回调：Selected points ====

@app.callback(
    Output("selected-points", "children"),
    Input("graph", "selectedData"),
)
def update_selected_points(selectedData):
    if not selectedData or "points" not in selectedData or len(selectedData["points"]) == 0:
        return (
            "Selected points: 0. No points selected. "
            "Use Box Select or Lasso Select on the graph toolbar to highlight points."
        )

    pts = selectedData["points"]
    rows = []
    for i, p in enumerate(pts, start=1):
        x_val = p.get("x", None)
        y_val = p.get("y", None)
        rows.append(

            html.Tr(
                [
                    html.Td(i),
                    html.Td(f"{x_val:.2f}" if x_val is not None else ""),
                    html.Td(f"{y_val:.2f}" if y_val is not None else ""),
                ]
            )
        )

    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("#"), html.Th("x"), html.Th("y")])),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    return [
        html.P(f"Selected points: {len(pts)}"),
        table,
    ]


# ==== 回调：函数历史记录 ====

@app.callback(
    Output("history-store", "data"),
    Output("history", "children"),
    Input("func-type", "value"),
    Input("slider-a", "value"),
    Input("slider-b", "value"),
    Input("slider-c", "value"),
    State("history-store", "data"),
)
def update_history(func_type, a, b, c, history):
    if history is None:
        history = []

    formula = build_formula(func_type, a, b, c)
    label = f"{FUNC_LABELS[func_type]}: {formula}"

    if len(history) == 0 or history[-1].get("label") != label:
        history.append({"label": label})

    history = history[-8:]  # 保留最近 8 条

    items = [html.Li(h["label"]) for h in history]
    if not items:
        children = "Recent functions you explored will appear here."
    else:
        children = html.Ul(items, style={"marginTop": 0})

    return history, children


# ==== 启动应用 ====

if __name__ == "__main__":
    app.run(debug=True)