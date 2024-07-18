import pandas as pd
import os
from yt_dlp import YoutubeDL
from youtubesearchpython import VideosSearch
import time

# Leer el archivo CSV
df = pd.read_csv("~/Desktop/bd2/BD2-Proyecto2/data/spotify_millsongdata_2000.csv")
# df = df.iloc[1001:2001]
artist_n_song = df.iloc[:, :2]

# Crear una carpeta para las descargas si no existe
download_folder = "songs"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Función para buscar y descargar una canción
def search_and_download(index, artist, song, download_folder):
    query = f"{artist} {song}"
    videos_search = VideosSearch(query, limit=1)
    results = videos_search.result()

    if results['result']:
        video_url = results['result'][0]['link']
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(download_folder, f'{index} - {artist} - {song}.%(ext)s'),
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',  # AAC
                'preferredquality': '64',
            }],
        }
        with YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
                return True
            except Exception as e:
                print(f"Error downloading {query}: {e}")
                return False
    else:
        print(f"No results found for {query}")
        return False

# Descargar cada canción
success_count = 0
for index, row in artist_n_song.iterrows():
    artist = row.iloc[0]
    song = row.iloc[1]
    success = search_and_download(index, artist, song, download_folder)
    if success:
        success_count += 1
    time.sleep(1)  # Para evitar hacer demasiadas solicitudes rápidamente

# Verificar que se hayan descargado todas las canciones
print(f"Successfully downloaded {success_count} out of {len(artist_n_song)} songs.")
