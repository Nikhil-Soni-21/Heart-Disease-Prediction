import dash
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table as dt
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import model_fun

data = pd.read_csv('assets/heart.csv')
print(model_fun.pred_naive_bayes(data=data))
graph_titles = [
    'Heart Disease Distribution',
    'Heart Disease Distribution By Gender',
    'Heart Disease Distribution By Chest Pain Type']


def sample_table():
    return dt.DataTable(
        id='sample_table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.sample(7).to_dict('records'),
        style_header={'fontWeight': 'bold'},
        style_cell={'padding': '8px', 'fontSize': '20px'},
        style_table={'width': '80%'}
    )


def get_graph(i, h):
    return px.histogram(
        data_frame=data,
        x='target',
        color=h,
        title=graph_titles[i],
        barmode='group',
        height=600,
        category_orders={'sex': [0, 1], 'cp': [0, 1, 2, 3], 'target': [0, 1]}
    )


def get_target_scatter():
    data['target'] = data['target'].astype(str)
    fig = px.scatter(
        data_frame=data,
        y=data.thalach,
        x=data.age,
        color='target',
        height=600,
        color_discrete_sequence=px.colors.qualitative.Set1,
        title='Maximum Heart Rate to Age Scatter Plot')
    fig.update_traces(marker=dict(size=10))
    return fig


app = dash.Dash(__name__)
app.title = 'Heart Disease Prediction'
app.layout = html.Div(children=[
    html.Center(
        children=[html.H1('Heart Disease Prediction using Machine Learning')]),
    html.Br(),
    html.H2('Data Visualization: '),
    html.H3('Sample Data:'),
    html.Br(),
    html.Center(children=[sample_table()]),
    html.Br(), html.Br(),
    html.H3('Visualizing Data: '),
    dcc.Dropdown(
        id='graph_selector',
        options=[{'label': i, 'value': i} for i in graph_titles],
        value=graph_titles[0],
        style={'width': '300px'}
    ),
    html.Br(),
    html.Div(id='graph_holder'),
    html.Br(),
    html.H3('Maximum Heart Rate to Age Scatter Plot:'),
    dcc.Graph(figure=get_target_scatter())
])


@app.callback(
    Output('graph_holder', 'children'),
    [Input('graph_selector', 'value')])
def change_graph(value):
    if value == graph_titles[0]:
        return dcc.Graph(figure=get_graph(0, 'target'))
    elif value == graph_titles[1]:
        return dcc.Graph(figure=get_graph(1, 'sex'))
    elif value == graph_titles[2]:
        return dcc.Graph(figure=get_graph(2, 'cp'))
    else:
        assert('Invalid Argument')


if __name__ == '__main__':
    app.run_server(debug=True)
