import numpy
from dash import Dash, dcc, html
import plotly.express as px
import csv
import ast
import statistics
import pandas as pd
import matplotlib as mpl
from base64 import b64encode
import io

app = Dash(__name__)

buffer = io.StringIO()

num_agents = 3
EPISODE_REWARD_MEANS = []
MEANS_EPISODE_REWARD_MEANS = []
with open(str(num_agents) + 'UAV_training_results.txt', 'r') as fd:
    [EPISODE_REWARD_MEANS.append(ast.literal_eval(line)) for line in fd.readlines()]

[MEANS_EPISODE_REWARD_MEANS.append(statistics.mean(e)) for e in EPISODE_REWARD_MEANS]

headers = [str(num_agents)+'UAV']
df = pd.DataFrame(MEANS_EPISODE_REWARD_MEANS, columns=headers)

min_num = 100
fig = px.line(df, title=str(num_agents)+"UAV").update_layout(xaxis_title="Checkpoints", yaxis_title="Recompensa acumulada")

fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

app.layout = html.Div([
    dcc.Graph(id="interactive-html-export-x-graph",
              style={"width": "100%", "display": "inline-block", "height": "600px"},
              figure=fig),
    html.A(
        html.Button("Download as HTML"),
        id="interactive-html-export-x-download",
        href="data:text/html;base64," + encoded,
        download="plotly_graph.html"
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
