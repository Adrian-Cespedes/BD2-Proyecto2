import gradio as gr
import pandas as pd
import os
from dbmanager import DataStoreManager

current_dir = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(current_dir, os.pardir, "data")
songs_path = os.path.join(current_dir, os.pardir, "songs")

# data_path = os.path.join(base_path, "df_total.csv")
data_path = os.path.join(base_path, "spotify_millsongdata_57650.csv")
mmdata_path = os.path.join(base_path, "NEW_reduced_features.csv")
pca_model_path = os.path.join(base_path, "NEW_pca_model.pkl")
# manager class
df_headers = ["Artist", "Song", "Lyrics", "Score"]
manager = DataStoreManager(data_path, df_headers, mmdata_path, pca_model_path, songs_path)

n_boxes = 10


def testFunc(audio, k, r):
    result, time_taken = manager.retrieve_media_knn(audio, k)
    # result -> (song_path + song_name, song_name, score)
    elementsVisible = []
    elementsHiddden = []
    for song_path, song_name, score in result:
        elementsVisible.append(gr.Markdown(value = f"### {song_name} - Distancia: {score:.4f}", visible=True))
        elementsVisible.append(gr.Audio(value = song_path, label=song_name, visible=True))

    for _ in range(n_boxes - len(result)):
        elementsHiddden.append(gr.Markdown(visible=False, value=""))
        elementsHiddden.append(gr.Audio(visible=False, label=""))

    elementsHiddden.append(gr.Markdown(value=f"### Execution time: {time_taken:.4f} seconds"))
    
    return elementsVisible + elementsHiddden


def executeQuery(query, k):
    result, time_taken = manager.retrieve(query, k)
    return result, f"### Execution time: {time_taken:.4f} seconds"


outputAudios = [] 

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
                topKLabel1 = gr.Slider(
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
                    topKLabel2 = gr.Number(
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

        with gr.Row():
            with gr.Column():
                for _ in range(n_boxes):
                    temp = gr.Markdown(visible=False)
                    outputAudios.append(temp)
                    temp = gr.Audio(type="filepath", visible=False)
                    outputAudios.append(temp)

    timeLabel = gr.Markdown("### Execution time: ... seconds")

    techniqueLabel1.input(
        fn=manager.set_active_store, inputs=[techniqueLabel1], outputs=[]
    )

    techniqueLabel2.input(
        fn=manager.set_active_store, inputs=[techniqueLabel2], outputs=[]
    )

    btn1.click(
        fn=executeQuery,
        inputs=[queryLabel, topKLabel1],
        outputs=[dfResult, timeLabel],
    )

    btn2.click(
        fn=testFunc,
        inputs=[audioInputLabel, topKLabel2, rLabel],
        # outputs=[dfResult, timeLabel],
        outputs=outputAudios + [timeLabel],
    )


demo.launch()
