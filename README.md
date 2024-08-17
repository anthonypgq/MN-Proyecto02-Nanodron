# Proyecto de Métodos Numéricos II Bimestre

# Simulación de Trajectoria de un Nanodron

## Descripción

Este proyecto simula la trayectoria de un nanodron con masa despreciable en meteorología utilizando el método de Euler para resolver las ecuaciones diferenciales de movimiento. El programa permite la entrada de constantes α, β, γ y condiciones iniciales, simula el movimiento durante un tiempo dado y grafica la trayectoria del nanodron.

## Instalación

### Instalar Dependencias

Asegúrate de tener instalado Python en tu sistema. Este script es compatible con Python 3.x. Puedes descargar e instalar Python desde [python.org](https://www.python.org/downloads/).

Para instalar las dependencias necesarias, sigue estos pasos:

1. Descarga el archivo `requirements.txt` del repositorio.
2. Abre una terminal o línea de comandos en el directorio donde se encuentra el archivo `requirements.txt`.
3. Ejecuta el siguiente comando para instalar las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

El archivo `requirements.txt` debe contener las siguientes dependencias:

matplotlib==3.9.2
numpy==2.0.1

## Uso del programa

1. **Ejecutar el Script**

    Para iniciar la simulación, ejecuta el archivo `nanodron_simulation.py` con el siguiente comando:

    ```bash
    python nanodron_simulation.py
    ```

2. **Entrada de Datos**

    El programa solicitará los siguientes datos para la simulación:
    - **Constantes α, β, γ**: Introduce los valores para las constantes que afectan el movimiento del nanodron.
    - **Condiciones Iniciales**: Proporciona las condiciones iniciales de posición y velocidad del nanodron.
    - **Tiempo de Simulación**: Especifica el tiempo total para la simulación.

3. **Ver Resultados**

    Después de ejecutar el script y proporcionar los datos, el programa realizará la simulación y generará gráficos que muestran la trayectoria del nanodron en función del tiempo.
