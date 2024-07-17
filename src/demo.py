import gradio as gr
import pandas as pd
import os
from dbmanager import DataStoreManager

current_dir = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(current_dir, os.pardir, "data")

# data_path = os.path.join(base_path, "df_total.csv")
data_path = os.path.join(base_path, "spotify_millsongdata_1000.csv")
mmdata_path = os.path.join(base_path, "reduced_features.csv")
pca_model_path = os.path.join(base_path, "pca_model.pkl")
# manager class
df_headers = ["Artist", "Song", "Lyrics", "Score"]
manager = DataStoreManager(data_path, df_headers, mmdata_path, pca_model_path)


def testFunc(audio, k, r):
    result, time_taken = manager.retrieve_media_knn(audio, k)
    print("result: ", result)
    return


def executeQuery(query, k):
    result, time_taken = manager.retrieve(query, k)
    return result, f"### Execution time: {time_taken} seconds"


with gr.Blocks(title="Proyecto 2") as demo:
    gr.Markdown(
        """
    # Proyecto 2
    """
    )
    
    with gr.Tab("Parte 1"):
        with gr.Row():
            with gr.Column(scale=2):
                queryLabel = gr.Textbox(label="Ingrese la consulta:", lines=1)
                btn1     = gr.Button("Ejecutar")
            with gr.Column(scale=1):
                topKLabel = gr.Slider(
                    label="Top K:", value=10, minimum=1, maximum=20, step=1, scale=0, interactive=True
                )
                techniqueLabel1 = gr.Dropdown(
                    label="Técnica:",
                    # choices=["Propia", "PostgreSQL", "MongoDB"],
                    choices=[
                        ("Propia", "inverted_index"),
                        # ("Propia RAM", "inverted_index_ram"),
                        ("MongoDB", "mongo"),
                    ],
                    value="inverted_index",
                    scale=0,
                    interactive=True
                )

        with gr.Row():
            # gr.Textbox(label="greeting", lines=3)
            dfResult = gr.Dataframe(
                label="Resultados:",
                headers=df_headers,
                row_count=(8, "dynamic"),
                height=400,
            )

    with gr.Tab("Parte 2"):
        with gr.Row():
            with gr.Column(scale=3):
                audioInputLabel = gr.Audio(label="Audio:", type="filepath")
                btn2 = gr.Button("Ejecutar")
            with gr.Column(scale=2):
                with gr.Row():
                    topKLabel = gr.Number(
                        label="Top K:", value=10, minimum=1, maximum=20, step=1, interactive=True
                    )
                    rLabel = gr.Number(
                        label="Rango:", value=0.5, step=0.1,    precision=2, interactive=True
                    )
                techniqueLabel2 = gr.Dropdown(
                    label="Técnica:",
                    choices=[
                        ("Secuencial", "knn_secuencial"),
                        ("RTree", "knn_rtree"),
                        ("Faiss", "knn_faiss"),
                    ],
                    value="knn_faiss",
                    scale=0,
                )


    timeLabel = gr.Markdown("### Execution time: ... seconds")

    techniqueLabel1.input(
        fn=manager.set_active_store, inputs=[techniqueLabel1], outputs=[]
    )

    techniqueLabel2.input(
        fn=manager.set_active_store, inputs=[techniqueLabel2], outputs=[]
    )

    btn1.click(
        fn=executeQuery,
        inputs=[queryLabel, topKLabel],
        outputs=[dfResult, timeLabel],
    )

    btn2.click(
        fn=testFunc,
        inputs=[audioInputLabel, topKLabel, rLabel],
        # outputs=[dfResult, timeLabel],
    )


demo.launch()
