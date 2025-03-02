import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42 # SEMILLA PARA GENERACIÓN DE NÚMEROS ALEATORIOS


# PARÁMETROS PARA SIMULACIONES
INTERVALS = [10, 5, 1]                    # Intervalos de llegada de procesos
PROCESS_COUNTS = [25, 50, 100, 150, 200]  # Lista de cantidad de procesos a simular
MEMORY_CAPACITY = [100, 200]              # Lista de capacidad de memoria a probar
CPU_SPEEDS = [3, 6]                       # Lista de velocidad del CPU 
CPU_COUNTS = [1, 2]                       # Lista de cantidad de CPUs a probar

# Variable global para la velocidad del CPU (se actualiza en correr_simulacion())
CPU_SPEED = 3  

def proceso(env, name, ram, cpu, instrucciones, resultados):
    """
    Función que modela el ciclo de vida de un proceso en el sistema operativo.
    Estados simulados: new, ready, running, waiting, terminated.
    
    Parámetros:
    - env: el entorno de simulación en SimPy
    - name: nombre del proceso
    - ram: contenedor de SimPy que modela la RAM
    - cpu: recurso de SimPy que modela la cola del CPU
    - instrucciones: cantidad de instrucciones totales a ejecutar
    - resultados: lista en la cual donde se almacena el tiempo total de cada proceso
    """
    
    # Momento de llegada del proceso
    llegada = env.now
    
    # Cantidad de memoria que necesita el proceso (entre 1 y 10)
    memoria = random.randint(1, 10)
    
    # NEW: solicitar memoria
    # Si no hay memoria suficiente, el proceso esperá en cola hasta que haya suficiente RAM 
    yield ram.get(memoria)
    
    while instrucciones > 0:

        # READY: solicitar acceso al CPU
        with cpu.request() as req:
            yield req

            # RUNNING: ejecutar la instruccion por 1 unidad de tiempo
            yield env.timeout(1)
            
            # El proceso ejecuta hasta CPU_SPEED  
            ejecutado = min(instrucciones, CPU_SPEED)
            instrucciones -= ejecutado
        
        # Decidir si el proceso continúa o hay I/O (waiting)
        if instrucciones > 0:

            # Generar un número entero entre 1 y 2 
            # Si es 1, el proceso pasa a Waiting; si es 2, regresa inmediatamente a Ready
            r = random.randint(1, 2)
            if r == 1:
                # WAITING: simular I/O
                yield env.timeout(1)
    
    # TERMINATED: liberar memoria
    ram.put(memoria)
    
    # Registrar el tiempo total 
    resultados.append(env.now - llegada)


def generar_procesos(env, interval, num_procesos, ram, cpu, resultados):
    """
    Genera procesos segun la distribución de llegadas y los introduce a la simulación
    
    Parámetros:
    - env: entorno de SimPy
    - interval: intervalo promedio de llegada de procesos
    - num_procesos: numero total de procesos 
    - ram: Container de SimPy para modelar RAM
    - cpu: Resource de SimPy para modelar CPU
    - resultados: lista de tiempos de cada proceso
    """
    for i in range(num_procesos):
        # Crear un proceso y agregarlo al entorno de SimPy
        env.process(proceso(env, f'Proceso-{i}', ram, cpu, random.randint(1, 10), resultados))
        # Esperar un tiempo aleatorio
        yield env.timeout(random.expovariate(1.0 / interval))


def correr_simulacion(interval, num_procesos, memory_capacity, cpu_speed, cpu_count):
    """
    Configura y corre la simulación para parámetros específicos.
    
    Parámetros:
    - interval: intervalo promedio de llegada de procesos
    - num_procesos: número total de procesos
    - memory_capacity: capacidad total de RAM (Container)
    - cpu_speed: velocidad del CPU 
    - cpu_count: cantidad de CPUs (Resource)
    
    Regresa:
    - promedio: tiempo promedio de un proceso en sistema
    - desviacion: desviación estándar del tiempo 
    """
    # Crear el entorno de SimPy
    env = simpy.Environment()
    
    # Crear Container del RAM
    ram = simpy.Container(env, init=memory_capacity, capacity=memory_capacity)
    
    # Crear Resource del CPU (capacidad 1 o 2)
    cpu = simpy.Resource(env, capacity=cpu_count)
    
    # Configurar la variable global CPU_SPEED
    global CPU_SPEED
    CPU_SPEED = cpu_speed
    
    # Lista para guardar los tiempos
    resultados = []
    
    # Iniciar proceso que generará a los procesos
    env.process(generar_procesos(env, interval, num_procesos, ram, cpu, resultados))
    
    # Ejecutar simulación hasta que TODOS los procesos terminen
    env.run()
    
    # Calcular y retornar el promedio y desviación estándar de los tiempos
    return np.mean(resultados), np.std(resultados)


def main():
    """
    Función principal que realiza las simulaciones con parámetros distintos y 
    genera gráficas del tiempo promedio
    """
    # Fijar la semilla para que los resultados sean constantes
    random.seed(RANDOM_SEED)
    
    # Probar todas las combinaciones de memoria, velocidad de CPU y cantidad de CPUs
    for memory in MEMORY_CAPACITY:
        for cpu_speed in CPU_SPEEDS:
            for cpu_count in CPU_COUNTS:
                plt.figure()
                
                for interval in INTERVALS:
                    tiempos = []
                    
                    for num_procesos in PROCESS_COUNTS:
                        promedio, _ = correr_simulacion(interval, num_procesos, memory, cpu_speed, cpu_count)
                        tiempos.append(promedio)
                    
                    plt.plot(PROCESS_COUNTS, tiempos, marker='o', label=f'Intervalo {interval}')
                
                plt.xlabel('Número de procesos')
                plt.ylabel('Tiempo promedio en el sistema')
                plt.title(f'Memoria: {memory}, CPU Velocidad: {cpu_speed}, CPUs: {cpu_count}')
                plt.legend()
                plt.show()


if __name__ == '__main__':
    main()
