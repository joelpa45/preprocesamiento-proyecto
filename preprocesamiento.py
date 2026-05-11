"""
Módulo de Preprocesamiento de Datos
Funciones para realizar el preprocesamiento completo de datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class PreprocessingPipeline:
    """
    Clase para gestionar el preprocesamiento completo de datasets
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
    
    def cargar_datos(self, ruta_archivo, separador=','):
        """
        Carga datos desde un archivo CSV
        
        Args:
            ruta_archivo (str): Ruta al archivo CSV
            separador (str): Separador del archivo
            
        Returns:
            DataFrame: Datos cargados
        """
        try:
            datos = pd.read_csv(ruta_archivo, sep=separador)
            print(f"✓ Datos cargados exitosamente: {datos.shape}")
            return datos
        except Exception as e:
            print(f"✗ Error al cargar datos: {e}")
            return None
    
    def explorar_datos(self, df):
        """
        Muestra información básica del dataset
        
        Args:
            df (DataFrame): Dataset a explorar
        """
        print("\n=== INFORMACIÓN DEL DATASET ===")
        print(f"Dimensiones: {df.shape}")
        print(f"\nTipos de datos:\n{df.dtypes}")
        print(f"\nValores nulos:\n{df.isnull().sum()}")
        print(f"\nEstadísticas descriptivas:\n{df.describe()}")
    
    def manejar_valores_nulos(self, df, estrategia='mean', columnas=None):
        """
        Maneja valores nulos en el dataset
        
        Args:
            df (DataFrame): Dataset
            estrategia (str): 'mean', 'median', 'most_frequent', 'constant'
            columnas (list): Columnas a imputar (None = todas numéricas)
            
        Returns:
            DataFrame: Dataset con valores imputados
        """
        df_copia = df.copy()
        
        if columnas is None:
            columnas = df_copia.select_dtypes(include=[np.number]).columns
        
        self.imputer = SimpleImputer(strategy=estrategia)
        df_copia[columnas] = self.imputer.fit_transform(df_copia[columnas])
        
        print(f"✓ Valores nulos manejados con estrategia '{estrategia}'")
        return df_copia
    
    def eliminar_duplicados(self, df):
        """
        Elimina filas duplicadas del dataset
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset sin duplicados
        """
        duplicados_antes = df.duplicated().sum()
        df_limpio = df.drop_duplicates()
        duplicados_eliminados = duplicados_antes - df_limpio.duplicated().sum()
        
        print(f"✓ Duplicados eliminados: {duplicados_eliminados}")
        return df_limpio
    
    def codificar_categoricas(self, df, columnas):
        """
        Codifica variables categóricas usando Label Encoding
        
        Args:
            df (DataFrame): Dataset
            columnas (list): Lista de columnas categóricas
            
        Returns:
            DataFrame: Dataset con variables codificadas
        """
        df_copia = df.copy()
        
        for columna in columnas:
            if columna in df_copia.columns:
                le = LabelEncoder()
                df_copia[columna] = le.fit_transform(df_copia[columna].astype(str))
                self.label_encoders[columna] = le
                print(f"✓ Columna '{columna}' codificada")
        
        return df_copia
    
    def normalizar_datos(self, df, columnas=None, metodo='standard'):
        """
        Normaliza o estandariza datos numéricos
        
        Args:
            df (DataFrame): Dataset
            columnas (list): Columnas a normalizar (None = todas numéricas)
            metodo (str): 'standard' o 'minmax'
            
        Returns:
            DataFrame: Dataset normalizado
        """
        df_copia = df.copy()
        
        if columnas is None:
            columnas = df_copia.select_dtypes(include=[np.number]).columns
        
        if metodo == 'standard':
            self.scaler = StandardScaler()
        elif metodo == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Método debe ser 'standard' o 'minmax'")
        
        df_copia[columnas] = self.scaler.fit_transform(df_copia[columnas])
        
        print(f"✓ Datos normalizados con método '{metodo}'")
        return df_copia
    
    def eliminar_outliers(self, df, columnas, metodo='iqr', umbral=1.5):
        """
        Elimina outliers usando el método IQR
        
        Args:
            df (DataFrame): Dataset
            columnas (list): Columnas a analizar
            metodo (str): Método de detección ('iqr' o 'zscore')
            umbral (float): Umbral para IQR (default: 1.5) o Z-score (default: 3)
            
        Returns:
            DataFrame: Dataset sin outliers
        """
        df_copia = df.copy()
        filas_antes = len(df_copia)
        
        for columna in columnas:
            if metodo == 'iqr':
                Q1 = df_copia[columna].quantile(0.25)
                Q3 = df_copia[columna].quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - umbral * IQR
                limite_superior = Q3 + umbral * IQR
                df_copia = df_copia[
                    (df_copia[columna] >= limite_inferior) & 
                    (df_copia[columna] <= limite_superior)
                ]
            elif metodo == 'zscore':
                z_scores = np.abs((df_copia[columna] - df_copia[columna].mean()) / df_copia[columna].std())
                df_copia = df_copia[z_scores < umbral]
        
        filas_eliminadas = filas_antes - len(df_copia)
        print(f"✓ Outliers eliminados: {filas_eliminadas} filas")
        
        return df_copia
    
    def guardar_datos(self, df, ruta_salida):
        """
        Guarda el dataset preprocesado
        
        Args:
            df (DataFrame): Dataset a guardar
            ruta_salida (str): Ruta del archivo de salida
        """
        try:
            df.to_csv(ruta_salida, index=False)
            print(f"✓ Datos guardados en: {ruta_salida}")
        except Exception as e:
            print(f"✗ Error al guardar datos: {e}")
    
    def pipeline_completo(self, ruta_entrada, ruta_salida, config):
        """
        Ejecuta el pipeline completo de preprocesamiento
        
        Args:
            ruta_entrada (str): Ruta del archivo de entrada
            ruta_salida (str): Ruta del archivo de salida
            config (dict): Configuración del preprocesamiento
            
        Returns:
            DataFrame: Dataset preprocesado
        """
        print("\n" + "="*50)
        print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*50 + "\n")
        
        # Cargar datos
        df = self.cargar_datos(ruta_entrada)
        if df is None:
            return None
        
        # Explorar datos
        if config.get('explorar', True):
            self.explorar_datos(df)
        
        # Eliminar duplicados
        if config.get('eliminar_duplicados', True):
            df = self.eliminar_duplicados(df)
        
        # Manejar valores nulos
        if config.get('manejar_nulos', True):
            df = self.manejar_valores_nulos(
                df, 
                estrategia=config.get('estrategia_nulos', 'mean')
            )
        
        # Codificar variables categóricas
        if 'columnas_categoricas' in config:
            df = self.codificar_categoricas(df, config['columnas_categoricas'])
        
        # Eliminar outliers
        if 'columnas_outliers' in config:
            df = self.eliminar_outliers(
                df, 
                config['columnas_outliers'],
                umbral=config.get('umbral_outliers', 1.5)
            )
        
        # Normalizar datos
        if config.get('normalizar', True):
            df = self.normalizar_datos(
                df,
                metodo=config.get('metodo_normalizacion', 'standard')
            )
        
        # Guardar datos
        self.guardar_datos(df, ruta_salida)
        
        print("\n" + "="*50)
        print("PREPROCESAMIENTO COMPLETADO")
        print("="*50 + "\n")
        
        return df


# Ejemplo de uso
if __name__ == "__main__":
    print("Módulo de preprocesamiento listo para usar")
