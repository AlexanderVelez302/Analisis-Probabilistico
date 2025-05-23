import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway


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




#2.2 Punto del taller



# Filtrar datos solo para la categoría más rentable y los tres estados
filtered_df = df[
    (df['Category'] == 'Office Supplies') &
    (df['State'].isin(['Illinois', 'New York', 'California']))
]

# Agrupar los datos por estado
illinois_profits = filtered_df[filtered_df['State'] == 'Illinois']['Profit']
newyork_profits = filtered_df[filtered_df['State'] == 'New York']['Profit']
california_profits = filtered_df[filtered_df['State'] == 'California']['Profit']

# Prueba ANOVA
anova_result = f_oneway(illinois_profits, newyork_profits, california_profits)
anova_result

#Resultado
f_onewayResult(statistic=1.156990686371407, pvalue=0.31634529680611473)

# Crear un gráfico boxplot de las ganancias por estado en la categoría "Office Supplies"
plt.figure(figsize=(8, 6))
sns.boxplot(x='State', y='Profit', data=filtered_df)
plt.title('Distribución de Ganancias por Estado - Office Supplies')
plt.xlabel('Estado')
plt.ylabel('Ganancia')
boxplot_path = "/mnt/data/boxplot_profit_by_state.png"
plt.tight_layout()
plt.savefig(boxplot_path)
plt.close()

# 2.3 Punto del taller  
from scipy.stats import kruskal

# Filtrar por estado más rentable y categoría más rentable
ny_df = df[(df['State'] == 'New York') & (df['Category'] == 'Office Supplies')]

# Agrupar ciudades con suficientes datos
ciudades_con_suficientes_datos = ny_df['City'].value_counts()
ciudades_filtradas = ciudades_con_suficientes_datos[ciudades_con_suficientes_datos >= 5].index  # al menos 5 registros
ny_df = ny_df[ny_df['City'].isin(ciudades_filtradas)]

# Agrupar profits por ciudad
profits_por_ciudad = [ny_df[ny_df['City'] == ciudad]['Profit'] for ciudad in ciudades_filtradas]

# Prueba de Kruskal-Wallis (no paramétrica)
kruskal_result = kruskal(*profits_por_ciudad)

print("\n--- Resultado de la prueba de Kruskal-Wallis ---")
print(f"Estadístico H: {kruskal_result.statistic:.4f}")
print(f"Valor p: {kruskal_result.pvalue:.4f}")

# Media de profit por ciudad
media_por_ciudad = ny_df.groupby('City')['Profit'].mean().sort_values(ascending=False)

print("\n--- Promedio de Profit por ciudad en New York ---")
print(media_por_ciudad)

print(f"\n✅ Ciudad más rentable: {media_por_ciudad.idxmax()} con un profit promedio de {media_por_ciudad.max():.2f}")

# 2.4 Punto del taller 

# Calcular rentabilidad total por categoría, estado y ciudad
category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
state_profit = df.groupby('State')['Profit'].sum().sort_values(ascending=False)
city_profit = df.groupby('City')['Profit'].sum().sort_values(ascending=False)

# Calcular promedios de ganancia por estado y ciudad
state_profit_mean = df.groupby('State')['Profit'].mean().sort_values(ascending=False)
city_profit_mean = df.groupby('City')['Profit'].mean().sort_values(ascending=False)

category_profit, state_profit.head(5), state_profit_mean.head(5), city_profit.head(5), city_profit_mean.head(5)