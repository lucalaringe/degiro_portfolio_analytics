from matplotlib.colors import to_rgb
import plotly.graph_objs as go

# House style
COLORS = {
    'indigo': '#6610f2',
    'green': '#4caf50',
    'yellow': '#ff9800',
    'blue': '#2196f3',
    'pink': '#e83e8c',
    'cyan': '#9c27b0',
    'red': '#e51c23',
    'black': '#333333',
    'orange': '#fd7e14',
    'purple': '#6f42c1',
    'teal': '#20c997',
    'gray': '#666',
}

# Plotly color_continuous_scale green, white, red
GRWHRE = [
    COLORS['red'],
    '#FFFFFF',
    COLORS['green']
]

FONT = 'Segoe UI'

# Plotly APG template
PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                showline=False,
                ticks=''
            ),
            angularaxis=dict(
                tickprefix='<b>',
                tickfont=dict(size=16),
            ),
        ),
        title=dict(font=dict(size=18, family=FONT), xanchor='center', x=0.5),
        font=dict(family=FONT, color=COLORS['black'], size=14),
        xaxis=dict(showline=False),
        yaxis=dict(showline=True, showspikes=True, zeroline=True),
        hoverlabel=dict(font=dict(family=FONT), align='left')
    ),
    data=dict(
        table=[dict(
            cells=dict(
                height=30
            )
        )]
    )
)


def opacitize(hex_color: str, alpha: float):
    """
    Function that converts a HEX color to
    an RGBA color with transparency.
    """
    rgb = to_rgb(hex_color)
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'


if __name__ == '__main__':
    pass
