import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

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
plt.ylabel('')  # Oculta etiqueta 
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

# Termination code point one

#Confidence Intervals point two

def calcular_intervalo_confianza(serie, nivel_confianza=0.95):
    serie = serie.dropna()
    n = len(serie)
    media = np.mean(serie)
    std = np.std(serie, ddof=1)
    alpha = 1 - nivel_confianza
    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margen_error = t * (std / np.sqrt(n))
    return media, media - margen_error, media + margen_error

categorias = df['Category'].unique()

for var in ['Amount', 'Profit', 'Quantity']:
    print(f"\n--- Intervalos de confianza para {var} por categoría ---")
    for cat in categorias:
        subset = df[df['Category'] == cat][var]
        media, lim_inf, lim_sup = calcular_intervalo_confianza(subset)
        print(f"{cat} - Media: {media:.2f}, IC 95%: [{lim_inf:.2f}, {lim_sup:.2f}]")

#Basándonos en la media y los intervalos de confianza de Profit,
# podemos considerar que la categoría más rentable es Office Supplies,
# aunque la diferencia con Electronics y Furniture no es excesiva.
#Esta información será clave para el siguiente paso: la prueba de hipótesis entre estados,
#  usando Office Supplies como categoría base.
    
#