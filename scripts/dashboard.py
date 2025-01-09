# dashboard.py

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# Load backtest results
df = pd.read_csv('backtest_results.csv')

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Backtest Results Dashboard'),

    html.Div(children='''
        Select a strategy to view detailed metrics.
    '''),

    dcc.Dropdown(
        id='strategy-dropdown',
        options=[{'label': row['Strategy'], 'value': row['Strategy']} for index, row in df.iterrows()],
        value=df['Strategy'].unique()[0]
    ),

    dcc.Graph(
        id='metrics-graph'
    ),

    dcc.Graph(
        id='return-graph'
    )
])

@app.callback(
    [dash.dependencies.Output('metrics-graph', 'figure'),
     dash.dependencies.Output('return-graph', 'figure')],
    [dash.dependencies.Input('strategy-dropdown', 'value')]
)
def update_graph(selected_strategy):
    filtered_df = df[df['Strategy'] == selected_strategy]
    
    # Metrics Bar Chart
    metrics = filtered_df[['Sharpe Ratio', 'Win Rate (%)', 'Profit Factor', 'Max Drawdown (%)']]
    metrics_melted = metrics.melt(var_name='Metric', value_name='Value')
    fig1 = px.bar(metrics_melted, x='Metric', y='Value', title=f"{selected_strategy} Metrics")
    
    # Total Return Line Chart
    fig2 = px.line(filtered_df, x=filtered_df.index, y='Total Return (%)', title=f"{selected_strategy} Total Return Over Parameters")
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True)

