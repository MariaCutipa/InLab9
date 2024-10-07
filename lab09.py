import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo de Excel subido para inspeccionar su contenido
file_path = 'BI_Postulantes09-1.xlsx'
excel_data = pd.ExcelFile(file_path)

# Verificar los nombres de las hojas para entender la estructura del archivo
excel_data.sheet_names

# Cargar los datos de 'Hoja1' en un DataFrame para análisis
df = pd.read_excel(file_path, sheet_name='Hoja1')

# Mostrar las primeras filas para entender su estructura
df.head()
# Imprimir
print(df)

##################################################
# Histogramas

# Ahora podemos aplicar el algoritmo K-Means. Primero, importemos las bibliotecas necesarias y seleccionemos las columnas relevantes para el clustering.

# Seleccionar columnas relevantes para el clustering, excluyendo las categóricas como 'Postulante' y 'Nom_Especialidad'
df_clustering = df[['Apertura Nuevos Conoc.', 'Nivel Organización', 'Participación Grupo Social', 
                    'Grado Empatía', 'Grado Nerviosismo', 'Dependencia Internet']]

# Aplicar el clustering KMeans con 3 clústeres como ejemplo (se puede ajustar)
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(df_clustering)

# Crear histogramas de diferentes variables en función de los clústeres y especialidades
plt.figure(figsize=(12, 8))

# Graficar histograma para 'Apertura Nuevos Conoc.' vs Clúster y Especialidad
sns.histplot(data=df, x='Apertura Nuevos Conoc.', hue='Cluster', multiple="stack", kde=True)
plt.title('Histograma: Apertura Nuevos Conocimientos vs Clústers')
plt.xlabel('Apertura Nuevos Conocimientos')
plt.ylabel('Frecuencia')
plt.show()

# Ahora generemos otro histograma que cruce 'Grado Empatía' con respecto a los Clústeres y Especialidad.
plt.figure(figsize=(12, 8))

# Graficar histograma para 'Grado Empatía' vs Clúster y Especialidad
sns.histplot(data=df, x='Grado Empatía', hue='Cluster', multiple="stack", kde=True)
plt.title('Histograma: Grado de Empatía vs Clústers')
plt.xlabel('Grado de Empatía')
plt.ylabel('Frecuencia')
plt.show()
