import matplotlib.pyplot as plt

# Datos de la tabla
palabras = [200, 400, 800, 1200, 1600, 2000]
sequential_times = [1.5411, 1.7626, 1.7271, 1.7480, 2.3369, 3.0378]
rtree_times = [1.2407, 1.2671, 1.3438, 1.6262, 1.4294, 1.3403]
faiss_times = [1.3904, 1.3060, 1.5598, 1.7280, 1.6997, 1.3089]

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(palabras, sequential_times, marker="o", label="Sequential KNN")
plt.plot(palabras, rtree_times, marker="o", label="RTree KNN")
plt.plot(palabras, faiss_times, marker="o", label="Faiss KNN")

# Configurar las etiquetas y el título
plt.xlabel("Número de Documentos (Canciones)")
plt.ylabel("Tiempo (segundos)")
plt.title("Comparación de Tiempos de Búsqueda (Log)")
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.xscale("log")  # Escala logarítmica para una mejor visualización de los datos
plt.yscale("log")
plt.show()
