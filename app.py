# +
from features import FeatureGenerator
from joblib import load
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go

model = load('xgb_linreg_rf.joblib')

feature_generator = FeatureGenerator()

def round_h(x):
    return np.round(2 * x) / 2

def score_essay(text):
    if not text:
        gr.Error("Empty essay text")
        return None, None
    df = pd.DataFrame({"full_text": [text]})
    df = feature_generator.generate_features(df)
    res = model.predict(df.iloc[:, 1:])[0].clip(1, 5)

    total_score = round_h(res.mean())
    rounded_res = round_h(res)
    predicted_values = ["cohesion", "conventions", "grammar", "phraseology", "syntax", "vocabulary"]
    res_df = pd.DataFrame(rounded_res, index=predicted_values)
    total_score_msg = str(total_score)
    if total_score > 4:
        total_score_msg += " \N{party popper}"
    return total_score_msg, go.Figure(go.Bar(x=rounded_res, y=predicted_values, orientation='h', text=rounded_res),
                     layout_xaxis_range=[1, 5])


examples = ['Hello, my name is Katherine, and I enjoy reading books. My favorite book is War and Peace by Leo Tolstoy.']
title = "English proficiency checker"
description =  "Your essay will be scored according to six analytic measures: cohesion, \
                syntax, vocabulary, phraseology, grammar, and conventions. \
                Each measure represents a component of proficiency in essay \
                writing, with greater scores corresponding to greater proficiency \
                in that measure. The scores range from 1.0 to 5.0 in increments of 0.5."

demo = gr.Interface(
    fn=score_essay,
    inputs=gr.Textbox(lines=2, show_label=False, placeholder="Put an essay here..."),
    outputs=[gr.Textbox(label='Your score', interactive=False), gr.Plot(show_label=False)],
    allow_flagging='never',
    examples=examples,
    title=title,
    description=description,
)
demo.launch()
