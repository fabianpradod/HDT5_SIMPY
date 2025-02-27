import simpy
import random
import numpy as np
import matplotlib.pyplot as plt


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
