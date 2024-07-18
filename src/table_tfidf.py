import matplotlib.pyplot as plt

# Datos de la tabla
palabras = [1000, 2000, 4000, 8000, 16000, 32000, 57650]
mongo_times = [0.0056, 0.0058, 0.0106, 0.0256, 0.0332, 0.0063, 0.0213]
inverted_index_times = [0.0427, 0.0787, 0.0915, 0.2028, 0.4128, 0.6036, 1.7574]

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(palabras, mongo_times, marker="o", label="MongoDB")
plt.plot(palabras, inverted_index_times, marker="o", label="Inverted Index")

# Configurar las etiquetas y el título
plt.xlabel("Número de Documentos (Canciones)")
plt.ylabel("Tiempo (segundos)")
plt.title("Comparación de Tiempos de Búsqueda")
plt.legend()

# Mostrar el gráfico
plt.grid(True)
# plt.xscale("log")  # Escala logarítmica para una mejor visualización de los datos
# plt.yscale("log")
plt.show()
