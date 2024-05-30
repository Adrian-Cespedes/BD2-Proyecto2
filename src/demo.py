import gradio as gr
import pandas as pd

# import numpy as np
import time
from inverseIndexRAM import test_func


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


def executeQuery(query, k, technique):
    st = time.time()
    # text = f"Query: {query}\nK: {k}\nTechnique: {technique}"

    # df = dummy_df()
    df = test_func(query, k)
    et = time.time()

    return df, f"### Execution time: {et - st} seconds"


with gr.Blocks(title="Proyecto 2") as demo:
    # df_headers = ["Score", "ID", "Title", "Description"]
    df_headers = ["ID", "Score"]

    gr.Markdown("""
    # Proyecto 2
    """)

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
                choices=["Propia", "PostgreSQL", "MongoDB"],
                value="Propia",
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

    btn.click(
        fn=executeQuery,
        inputs=[queryLabel, topKLabel, techniqueLabel],
        outputs=[dfResult, timeLabel],
    )


demo.launch()
