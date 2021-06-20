import dash
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.metrics import accuracy_score
import dash_table as dt
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from predict_model import ModelLoader

data = pd.read_csv('assets/heart.csv')
loader = ModelLoader(data)

graph_titles = [
    'Heart Disease Distribution',
    'Heart Disease Distribution By Gender',
    'Heart Disease Distribution By Chest Pain Type']

model_titles = [
    "Naive Bayes",
    "Logistic Regression",
    "SVM(Support Vector Machine)",
    "K Nearest Neighbour",
    "Decision Tree",
    "Random Forest",
    "XG Boost",
    "Neural Network"]


model_result = [
    loader.model(0),
    loader.model(1),
    loader.model(2),
    loader.model(3),
    loader.model(4),
    loader.model(5),
    loader.model(6),
    loader.neunet()
]


def get_accuracy(index) -> float:
    x = model_result[index]
    return round(accuracy_score(x, loader.y_test)*100, 2)


model_accuracy = [get_accuracy(i) for i in range(8)]


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
        style={'width': '320px'},
        clearable=False
    ),
    html.Br(),
    html.Div(id='graph_holder'),
    html.Br(),
    html.H3('Maximum Heart Rate to Age Scatter Plot:'),
    dcc.Graph(figure=get_target_scatter()),
    html.Br(), html.Br(),
    html.H2("Comparative Analysis"),
    html.Br(),
    dcc.Graph(figure=px.bar(x=model_titles, y=model_accuracy))
])


@ app.callback(
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
