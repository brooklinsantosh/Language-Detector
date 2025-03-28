import numpy as np
import pickle
import plotly.graph_objects as go

from src.modelling import Model
import src.const as const

class Detect():
    def __init__(self):
        pass

    def load_model(self):
        file = open(const.MODEL_FILE, 'rb')
        model = pickle.load(file)
        file.close()
        return model

    def detect(self, text):
        mdl = Model()
        text = mdl.preprocessor(text)
        model = self.load_model()
        lang = model.predict([text])[0]
        idx = np.argmax(model.predict_proba([text]))
        fig = go.Figure(go.Indicator(
            domain = {'x': [0,1], 'y':[0,1]},
            value = model.predict_proba([text])[0][idx]*100,
            mode = "gauge+number",
            title = {'text': 'Confidence %'},
            gauge = {'axis': {'range': [None, 100]},
                    'steps':[
                        {'range': [0,33], 'color': 'red'},
                        {'range': [33,66], 'color': 'orange'},
                        {'range': [66,100], 'color': 'green'}],
                    'bar': {'color': '#fafb7e'}
                    }
        ))
        fig.update_layout(
            autosize = False,
            width = 300,
            height = 250,
            margin = dict(
                l = 50,
                r = 50,
                b = 1,
                t = 1,
                pad =4
            ),
        )
        return lang, fig