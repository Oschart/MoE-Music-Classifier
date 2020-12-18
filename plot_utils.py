import os
import pathlib
import pickle
import warnings
from pathlib import Path
import plotly.figure_factory as ff
import cupy as cp
import numpy as np
import plotly.graph_objects as go
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

warnings.filterwarnings('ignore')


def plot_confusion_matrix(probs, classes):
    z_text = [[str(y) for y in x] for x in probs]
    # set up figure
    fig = ff.create_annotated_heatmap(
        probs, x=classes, y=classes, annotation_text=z_text, colorscale='burg')

    # add title
    fig.update_layout(title_text='<i><b>Normalized Confusion matrix</b></i>',
                      title_x=0.5
                      #xaxis = dict(title='x'),
                      #yaxis = dict(title='x')
                      )

    fig.update_yaxes(autorange="reversed")
    fig['layout']['xaxis'].update(side='bottom')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.2,
                            showarrow=False,
                            text="Predicted Genre",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.2,
                            y=0.5,
                            showarrow=False,
                            text="True Genre",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    # adjust margins to make room for yaxis title

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


def plot_loss_vs_epochs(loss_train, loss_val, epochs, title):

    fig = go.Figure(data=go.Scatter(
        x=epochs,
        y=loss_train,
        name="Training"
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_val,
        name="Validation"
    ))

    x_best = np.argmin(loss_val)
    y_best = loss_val[x_best]
    fig = add_arrow_annotation(
        fig, x_best, y_best, text='Stop here: Overfitting!')

    fig.add_shape(type="line",
                  x0=x_best, x1=x_best,
                  line=dict(color="slategray", width=2))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        yaxis_title="Loss",
        xaxis_title="Epochs"
    )
    # fig.update_xaxes(type="category")
    #fig.write_html('%s/%s_interactive.html' % (plot_dir, fname))
    fig.show()


def plot_tuning_results(params, val_accrs, epochs, param_name):

    fig = go.Figure()

    for i, v in enumerate(params):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=cp.asnumpy(val_accrs[i]),
            name=param_name+" = "+str(v),
        ))

    fig.update_layout(
        title_text=param_name + " Fine-tuning",
        title_x=0.5,
        yaxis_title="Validation Loss",
        xaxis_title='Epochs'
    )
    # fig.update_xaxes(type="category")
    fig.show()


def add_arrow_annotation(fig, x, y, text=''):
    fig.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#000000",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#008000",
        opacity=0.8
    )
    return fig
