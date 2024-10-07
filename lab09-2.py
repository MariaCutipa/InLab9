import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Cargar los datos del archivo de Excel subido
file_path_clients = 'BI_Clientes09-1.xlsx'
df_clients = pd.read_excel(file_path_clients, sheet_name='Hoja1')

# Seleccionar columnas relevantes para el modelo de árbol de decisiones
df_clients_filtered = df_clients[['Age', 'YearOfFirstPurchase', 'CommuteDistance', 'BikeBuyer']].dropna()

# Convertir 'CommuteDistance' a valores numéricos para el modelado
df_clients_filtered['CommuteDistance'] = df_clients_filtered['CommuteDistance'].map({
    '0-1 Miles': 1, '2-5 Miles': 2, '5-10 Miles': 3, '10+ Miles': 4
})

# Manejo de valores faltantes eliminando filas con NaNs
df_clients_filtered_cleaned = df_clients_filtered.dropna()

# Imprimir
print(df_clients)

# Dividir el conjunto de datos limpio en características y objetivo
X_cleaned = df_clients_filtered_cleaned[['Age', 'YearOfFirstPurchase', 'CommuteDistance']]
y_cleaned = df_clients_filtered_cleaned['BikeBuyer']

# Dividir en conjuntos de entrenamiento y prueba
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=0)

# Inicializar y entrenar el clasificador de árbol de decisiones
clf_cleaned = DecisionTreeClassifier(random_state=0)
clf_cleaned = clf_cleaned.fit(X_train_cleaned, y_train_cleaned)

# Graficar el árbol de decisiones
plt.figure(figsize=(16, 10))
tree.plot_tree(clf_cleaned, feature_names=['Age', 'YearOfFirstPurchase', 'CommuteDistance'], 
               class_names=['No Comprador', 'Comprador'], filled=True)
plt.title('Árbol de Decisiones: Predicción de Compra de Bicicletas')
plt.show()
