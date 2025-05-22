import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('../data/sales_dataset.csv')


# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Verificar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Mostrar información general del dataset
print("\nInformación del dataset:")
print(df.info())

print("\n--- Análisis estadístico descriptivo ---")
print(df[['Amount', 'Profit', 'Quantity']].describe())

print("\n--- Total de ventas por mes ---")
ventas_mensuales = df.groupby('Year-Month')['Amount'].sum()
print(ventas_mensuales)

ventas_mensuales.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Ventas Totales por Mes')
plt.xlabel('Mes')
plt.ylabel('Ventas (Amount)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ventas_por_categoria = df.groupby('Category')['Amount'].sum()
print("\n--- Ventas por categoría ---")
print(ventas_por_categoria)

ventas_por_categoria.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Distribución de Ventas por Categoría')
plt.ylabel('')  # Oculta etiqueta fea
plt.show()

top_ciudades = df.groupby('City')['Amount'].sum().sort_values(ascending=False).head(5)
print("\n--- Top 5 ciudades con más ventas ---")
print(top_ciudades)

top_ciudades.plot(kind='bar', color='orange')
plt.title('Top 5 Ciudades con Más Ventas')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
