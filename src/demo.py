import gradio as gr
import pandas as pd
import os
from dbmanager import DataStoreManager

# import numpy as np


current_dir = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(current_dir, os.pardir, "data")

# data_path = os.path.join(base_path, "df_total.csv")
data_path = os.path.join(base_path, "spotify_millsongdata_16000.csv")
# manager class
df_headers = ["Artist", "Song", "Lyrics", "Score"]
manager = DataStoreManager(data_path, df_headers)


def dummy_df():
    dummydf = pd.DataFrame(
        {
            "Score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1],
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "Title": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
            "Description": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
        }
    )
    return dummydf


def executeQuery(query, k):
    # text = f"Query: {query}\nK: {k}\nTechnique: {technique}"

    # df = dummy_df()
    # df = test_func(query, k)
    result, time_taken = manager.retrieve(query, k)

    return result, f"### Execution time: {time_taken} seconds"


with gr.Blocks(title="Proyecto 2") as demo:
    # df_headers = ["Score", "ID", "Title", "Description"]
    # df_headers = ["ID", "Score"]

    gr.Markdown(
        """
    # Proyecto 2
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            queryLabel = gr.Textbox(label="Ingrese la consulta:", lines=1)
            btn = gr.Button("Ejecutar")
        with gr.Column(scale=1):
            topKLabel = gr.Slider(
                label="Top K:", value=10, minimum=1, maximum=20, step=1, scale=0
            )
            techniqueLabel = gr.Dropdown(
                label="Técnica:",
                # choices=["Propia", "PostgreSQL", "MongoDB"],
                choices=[
                    ("Propia", "inverted_index"),
                    # ("Propia RAM", "inverted_index_ram"),
                    ("MongoDB", "mongo"),
                ],
                value="inverted_index",
                scale=0,
            )

    with gr.Row():
        # gr.Textbox(label="greeting", lines=3)
        dfResult = gr.Dataframe(
            label="Resultados:",
            headers=df_headers,
            row_count=(8, "dynamic"),
            height=400,
        )

    timeLabel = gr.Markdown("### Execution time: ... seconds")

    techniqueLabel.input(
        fn=manager.set_active_store, inputs=[techniqueLabel], outputs=[]
    )

    btn.click(
        fn=executeQuery,
        inputs=[queryLabel, topKLabel],
        outputs=[dfResult, timeLabel],
    )


demo.launch()
