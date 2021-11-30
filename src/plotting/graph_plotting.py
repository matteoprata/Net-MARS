
import plotly.io as pio
import plotly.graph_objects as go
import src.constants as co
import numpy as np

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from src.preprocessing.graph_utils import *


def hovering_info_edges(n1, n2, id, capacity, post, wight=None, state=None, sat=None):
    sat = dict(sat) if sat is not None else sat
    return "<br><b>e({},{})({})</b> <br>- cap: {} <br>- post broken: {} <br>- weight: {} <br>- state_T: {} <br>- sat:{}".format(n1, n2, id, capacity, post, wight, state, sat)


def hovering_info_nodes(n1, id, post, state):
    return "<br><b>v({})({})</b> <br>- post broken: {} <br>- state_T: {}".format(n1, id, post, state)


def node_trace_make(G, scale_visual, density, plot_type, scalar_map1, scalar_map2, demand_edges):
    node_x = []
    node_y = []

    x_density = scale_visual["x"]*density
    y_density = scale_visual["x"]*density

    for node in G.nodes():
        x, y = G.nodes[node][co.ElemAttr.LONGITUDE.value]*x_density, G.nodes[node][co.ElemAttr.LATITUDE.value]*y_density
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            line_color="black",
            size=10,
            line_width=1))

    node_color = []
    node_text = []
    for node in G.nodes:

        prob = round(G.nodes[node][co.ElemAttr.POSTERIOR_BROKEN.value], 3)
        state_T = G.nodes[node][co.ElemAttr.STATE_TRUTH.value]
        ide = G.nodes[node][co.ElemAttr.ID.value]

        text = hovering_info_nodes(node, ide, prob, state_T)
        node_text.append(text)

        if plot_type == co.PlotType.TRU:
            probability = 1 - G.nodes[node][co.ElemAttr.STATE_TRUTH.value]
            color_rgb = scalar_map1.to_rgba(probability)
            color = ('rgb(%4.2f,%4.2f,%4.2f)' % (color_rgb[0], color_rgb[1], color_rgb[2]))
        elif plot_type == co.PlotType.KNO:
            probability = 1 - G.nodes[node][co.ElemAttr.POSTERIOR_BROKEN.value]
            color_rgb = scalar_map1.to_rgba(probability)
            color = ('rgb(%4.2f,%4.2f,%4.2f)' % (color_rgb[0], color_rgb[1], color_rgb[2]))
        else:
            color = "yellow"

        node_color.append(color)

    node_trace.marker.color = node_color
    node_trace.text = node_text
    return node_trace


def edge_trace_make(G, scale_visual, density, plot_type, scalar_map1, scalar_map2, demand_edges):
    edge_traces = []

    x_density = scale_visual["x"]*density
    y_density = scale_visual["x"]*density

    for n1, n2, gt_ori in G.edges:
        edge_x = []
        edge_y = []

        x0, y0 = G.nodes[n1][co.ElemAttr.LONGITUDE.value]*x_density, G.nodes[n1][co.ElemAttr.LATITUDE.value]*y_density
        x1, y1 = G.nodes[n2][co.ElemAttr.LONGITUDE.value]*x_density, G.nodes[n2][co.ElemAttr.LATITUDE.value]*y_density

        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

        capacity = str(round(G.edges[n1, n2, gt_ori][co.ElemAttr.RESIDUAL_CAPACITY.value], 3)) + " / " + \
                   str(round(G.edges[n1, n2, gt_ori][co.ElemAttr.CAPACITY.value], 3))

        if gt_ori == co.EdgeType.DEMAND.value:  # demand edge
            prob = round(G.edges[n1, n2, gt_ori][co.ElemAttr.RESIDUAL_CAPACITY.value], 3)
            sat = G.edges[n1, n2, gt_ori][co.ElemAttr.SAT_SUP]
            text = hovering_info_edges(n1, n2, 'D', capacity, prob, sat=sat)
            dash = 'dash'
            color = 'blue'
        else:
            dash = None
            prob = round(G.edges[n1, n2, gt_ori][co.ElemAttr.POSTERIOR_BROKEN.value], 3)
            weight = round(G.edges[n1, n2, gt_ori][co.ElemAttr.WEIGHT.value], 3)
            state_T = G.edges[n1, n2, gt_ori][co.ElemAttr.STATE_TRUTH.value]
            sat = G.edges[n1, n2, gt_ori][co.ElemAttr.SAT_DEM.value]
            ide = G.edges[n1, n2, gt_ori][co.ElemAttr.ID.value]
            text = hovering_info_edges(n1, n2, ide, capacity, prob, weight, state_T, sat)

            if plot_type == co.PlotType.TRU:
                probability = 1 - G.edges[n1, n2, gt_ori][co.ElemAttr.STATE_TRUTH.value]
            elif plot_type == co.PlotType.KNO:
                probability = 1-G.edges[n1, n2, gt_ori][co.ElemAttr.POSTERIOR_BROKEN.value]
            else:
                probability = 1-G.edges[n1, n2, gt_ori][co.ElemAttr.POSTERIOR_BROKEN.value]

            color_rgb = scalar_map1.to_rgba(probability)
            color = ('rgb(%4.2f,%4.2f,%4.2f)' % (color_rgb[0], color_rgb[1], color_rgb[2]))

            # Here we color the edges colored based on their routing
            n_shared = len(G.edges[n1, n2, gt_ori][co.ElemAttr.SAT_DEM.value].keys())
            if plot_type == co.PlotType.ROU and n_shared > 0:
                shades, weight = [], []
                for d1, d2 in G.edges[n1, n2, gt_ori][co.ElemAttr.SAT_DEM.value].keys():
                    perc = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.SAT_SUP][(n1, n2)]
                    id = demand_edges.index((d1, d2))
                    color_rgb = scalar_map2.to_rgba(id)
                    shades.append([color_rgb[0], color_rgb[1], color_rgb[2]])
                    weight.append([perc])

                shades, weight = np.array(shades), (np.array(weight) if n_shared > 1 else np.array([1]))
                color_rgb = np.sum(np.multiply(shades, weight), axis=0)
                color = ('rgb(%4.2f,%4.2f,%4.2f)' % (color_rgb[0], color_rgb[1], color_rgb[2]))
            elif plot_type == co.PlotType.ROU and n_shared == 0:
                color = 'yellow'

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color=color, dash=dash),
            mode='lines'
        )

        middle_nodes_trace = middle_node_tracer(text, color, x0, x1, y0, y1)
        edge_traces.append(middle_nodes_trace)

        edge_traces.append(edge_trace)

    return edge_traces


def middle_node_tracer(text, color, x0, x1, y0, y1):
    """ Trace of the edge information node. It's at the middle. """
    middle_node_trace = go.Scatter(
        x=[(x0 + x1) / 2],
        y=[(y0 + y1) / 2],
        text=[text],
        mode='markers',
        hoverinfo='text',

        marker=dict(
            color=color,
            opacity=0
        )
    )
    return middle_node_trace


def setup_color_maps(n_demand_edges):
    cm = plt.get_cmap('RdYlGn')  # Wistia
    scale = colors.Normalize(vmin=0, vmax=1)
    scalar_map1 = cmx.ScalarMappable(norm=scale, cmap=cm)

    cm = plt.get_cmap('tab20')
    scale = colors.Normalize(vmin=0, vmax=n_demand_edges-1)
    scalar_map2 = cmx.ScalarMappable(norm=scale, cmap=cm)
    return scalar_map1, scalar_map2


def plot(G, graph_name, distribution, density, scale_visual, is_show, is_save, seed, name, plot_type):
    demand_edges = get_demand_edges(G, is_capacity=False)

    scalar_map1, scalar_map2 = setup_color_maps(len(demand_edges))
    edge_trace = edge_trace_make(G, scale_visual, density, plot_type, scalar_map1, scalar_map2, demand_edges)
    node_trace = node_trace_make(G, scale_visual, density, plot_type, scalar_map1, scalar_map2, demand_edges)

    heat_trace = []
    if plot_type == co.PlotType.TRU and distribution is not None:
        heat_trace = [go.Heatmap(z=distribution, opacity=0.2, showscale=False, hoverinfo='none')]

    fig = go.Figure(data=heat_trace + edge_trace + [node_trace],
                    layout=go.Layout(
                    plot_bgcolor='rgba(245,245,245,1)',
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, showline=True, mirror=True, ticks='outside', linecolor='black'),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, showline=True, mirror=True, ticks='outside', linecolor='black'))
                    )

    save_show_fig(fig, is_show, is_save, seed, graph_name, name, plot_type)


def save_show_fig(fig, is_show, is_save, seed, graph_name, name, plot_type):
    if is_show:
        fig.show()

    if is_save:
        dir = "data/dis_image/{}-{}-{}-{}".format(plot_type.name, seed, graph_name, name)
        fig.write_image(dir + ".png", width=1400, height=1120, scale=2)
        pio.write_html(fig, dir + ".html")
