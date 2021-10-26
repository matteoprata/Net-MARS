
import plotly.graph_objects as go
import src.constants as co
import numpy as np


def node_trace_make(G, scale_visual, density):
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
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: {}\npos_x: {}\npos_y: {}'.format(len(adjacencies[1]), node_x[node], node_y[node]))

    for node in G.nodes:
        if G.nodes[node]['state'] == co.NodeState.BROKEN.name:
            node_color.append(co.NodeState.BROKEN.value)
        elif G.nodes[node]['state'] == co.NodeState.WORKING.name:
            node_color.append(co.NodeState.WORKING.value)
        elif G.nodes[node]['state'] == co.NodeState.UNK.name:
            node_color.append(co.NodeState.UNK.value)
        else:
            exit("Un handled state.")

    # node_trace.marker.color = node_adjacencies
    node_trace.marker.color = node_color
    node_trace.text = node_text
    return node_trace


def edge_trace_make(G, scale_visual, density):
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

        color = None
        if G.edges[n1, n2, gt_ori]['state'] == co.NodeState.BROKEN.name:
            color = co.NodeState.BROKEN.value
        elif G.edges[n1, n2, gt_ori]['state'] == co.NodeState.WORKING.name:
            color = co.NodeState.WORKING.value
        elif G.edges[n1, n2, gt_ori]['state'] == co.NodeState.UNK.name:
            color = co.NodeState.UNK.value
        elif G.edges[n1, n2, gt_ori]['state'] == co.NodeState.NA.name:
            color = co.NodeState.NA.value
        else:
            exit("Un handled state.")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color=color),
            hoverinfo='none',
            mode='lines'
        )

        edge_traces.append(edge_trace)
    return edge_traces


def plot(G, graph_name, distribution, density, scale_visual, is_show, is_save, seed):

    edge_trace = edge_trace_make(G, scale_visual, density)
    node_trace = node_trace_make(G, scale_visual, density)

    heat_trace = []
    if distribution is not None:
        # distribution = augment_dims_for_fancy_plot(distribution)
        heat_trace = [go.Heatmap(z=distribution, opacity=0.4)]

    fig = go.Figure(data= heat_trace + edge_trace + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                    )
    if is_show:
        fig.show()

    if is_save:
        fig.write_image("data/dis_image/dis-{}-{}.pdf".format(seed, graph_name), width=1400, height=1120, scale=2)

    return fig





