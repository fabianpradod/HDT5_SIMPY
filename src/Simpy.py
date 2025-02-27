import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
INTERVALS = [10, 5, 1]
PROCESS_COUNTS = [25, 50, 100, 150, 200]
MEMORY_CAPACITY = [100, 200]  # Se probarán ambas capacidades
CPU_SPEEDS = [3, 6]  # Se probarán ambas velocidades
CPU_COUNTS = [1, 2]  # Se probarán 1 y 2 CPUs

def proceso(env, name, ram, cpu, instrucciones, resultados):
    llegada = env.now
    memoria = random.randint(1, 10)
    
    yield ram.get(memoria)
    
    with cpu.request() as req:
        yield req
        while instrucciones > 0:
            yield env.timeout(1)
            instrucciones -= min(instrucciones, CPU_SPEED)
    
    ram.put(memoria)
    resultados.append(env.now - llegada)


def generar_procesos(env, interval, num_procesos, ram, cpu, resultados):
    for i in range(num_procesos):
        env.process(proceso(env, f'Proceso-{i}', ram, cpu, random.randint(1, 10), resultados))
        yield env.timeout(random.expovariate(1.0 / interval))

def correr_simulacion(interval, num_procesos, memory_capacity, cpu_speed, cpu_count):
    env = simpy.Environment()
    ram = simpy.Container(env, init=memory_capacity, capacity=memory_capacity)
    cpu = simpy.Resource(env, capacity=cpu_count)
    global CPU_SPEED
    CPU_SPEED = cpu_speed
    resultados = []
    
    env.process(generar_procesos(env, interval, num_procesos, ram, cpu, resultados))
    env.run()
    return np.mean(resultados), np.std(resultados)
