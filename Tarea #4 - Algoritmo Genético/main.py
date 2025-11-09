import random, math, csv, ast, statistics

# ✅
def input_options(var, options, text ):
    while (not str(var).isnumeric() or int(var) not in options):
        var = input(text)
    return int(var)

# ✅
def input_natural_n(var, text):
    while (not str(var).isnumeric() or int(var) < 1):
        var = input(text)
    return int(var)

# ✅
def input_positive_f(var, text):
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    while (not isfloat(str(var)) or float(var) <= 0):
        var = input(text)
    return float(var)

# ✅
def guardar_poblacion(poblacion):
    while(True):
        file_name = input("Ingrese el nombre del archivo. (No incluir '.csv')\n -> ")
        data_rows = poblacion
        with open(f"./poblaciones/{file_name}.csv", mode='w', newline='') as file:
            csv.writer(file).writerow(data_rows)
        return 

# ✅
def cargar_poblacion():
    while(True):
        try:
            q="0"
            q = input_options(q, (1,2,3,4,5), "(1)La población entrenada con la baja variabilidad sobre la función ‘a’\n" \
            "(2)La población entrenada con la alta variabilidad sobre la función ‘a’\n" \
            "(3)La población entrenada con la baja variabilidad sobre la función ‘b’\n" \
            "(4)La población entrenada con la alta variabilidad sobre la función ‘b’\n" \
            "(5)Otra\n" \
            "-> ")
            match q:
                case 1:
                    file_name = "a1"
                case 2:
                    file_name = "a2"
                case 3:
                    file_name = "b1"
                case 4:
                    file_name = "b2"
                case 5:
                    file_name = input("Ingrese el nombre del archivo. (No incluir '.pth')\n -> ")
            
            with open(f"./poblaciones/{file_name}.csv") as file:
                data = csv.reader(file)
                for index, row in enumerate(data):
                    # [0:Tamaño, 1:Umbral de diferencia, 2: Maximo de generaciones, 3: Variabilidad de mutación, 4: Soluciones, 5: Tipo de población, 6: Cantidad de veces optimizada]
                    poblacion = [int(row[0]), float(row[1]), int(row[2]), float(row[3]), ast.literal_eval(row[4]), int(row[5]), int(row[6]) ]
                    return poblacion
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")
            return False

# ✅
def modificar_constantes():
    paciencia = prob_mutacion = tasa_supervivencia = "0"
    paciencia = input_positive_f(paciencia, "Paciencia (e.g. 0.25) -> ")
    prob_mutacion = input_positive_f(prob_mutacion, "Probabiliad de mutación (e.g. 0.05) -> ")
    tasa_supervivencia = input_positive_f(tasa_supervivencia, "Tasa de supervivencia (e.g. 0.25)-> ")
    return paciencia, prob_mutacion, tasa_supervivencia

# ✅
def fitness_a(x):
        if x == 0: return 0.0 # Devuelve 0 si x es 0 para evitar error
        return math.pow(math.sin(x), 2) / x

# ✅
def fitness_b(x,y):
    return 20 + x - 10*math.cos(2*math.pi*x) + y - 10*math.cos(2*math.pi*y)

# ✅
def algoritmo_evolutivo(poblacion, constantes, funcion):
    # CONSTANTES
    TAMAÑO = poblacion[0]
    UMBRAL_DIFERENCIA = poblacion[1]
    GENERACIONES_MAXIMAS = poblacion[2]
    VARIABILIDAD_MUTACION = poblacion[3]
    TASA_SUPERVIVENCIA = int(TAMAÑO*constantes[2]) # Seleccion Agresiva del 25%
    HIJOS_GEN = TAMAÑO - TASA_SUPERVIVENCIA
    PROB_MUTACION = constantes[1]
    PACIENCIA = int(GENERACIONES_MAXIMAS*constantes[0]) # Cuántas generaciones sin mejoras esperamos. 25% del total.
    soluciones = poblacion[4]
    func_text = ["y = ((sinx)^2) / x", "z = 20 + x - 10Cos(2pi*x) + y - 10Cos(2pi*y)"]

    match funcion:
        case 1: # [ y = ((sinx)^2) / x ]
            best_fitness_anterior = -math.inf 
            generacion_parada = 0 # Para el reporte final
            for gen in range(GENERACIONES_MAXIMAS):
                generacion_parada += 1
                # Evaluamos la población.
                eval_sol = []
                for x in soluciones: eval_sol.append( (x, fitness_a(x)) )

                # Ordenamos de mayor a menor
                eval_sol.sort(key= lambda x: x[1], reverse=True)

                # Info:
                fitness_value = [tupla[1] for tupla in eval_sol]
                max_fitness = fitness_value[0]   # El primero (ya está ordenado)
                min_fitness = fitness_value[-1]  # El último (ya está ordenado)
                median_fitness = statistics.median(fitness_value)
                print(f"Generación: {gen+1} | Máximo: {max_fitness:.6f} | Mediana: {median_fitness:.6f} | Mínimo: {min_fitness:.6f}")

                #Chequeamos el umbral de diferencia
                diferencia = max_fitness - best_fitness_anterior
                # Si la mejora es menor que el umbral, paramos
                if diferencia > UMBRAL_DIFERENCIA:
                    # ¡Hubo mejora significativa!
                    best_fitness_anterior = max_fitness
                    generaciones_sin_mejora = 0         # Reseteamos la paciencia
                else:
                    # No hubo mejora significativa
                    generaciones_sin_mejora += 1        # Incrementamos la paciencia
                
                # Comprobamos si se nos acabó la paciencia
                if generaciones_sin_mejora >= PACIENCIA:
                    print(f"--- Convergencia por estancamiento en Gen {gen+1} ---")
                    print(f"No hubo mejora > {UMBRAL_DIFERENCIA} en {PACIENCIA} generaciones. Deteniendo.")
                    break # Salir del bucle 'for gen'                

                # Seleccionamos padres con elitismo.
                padres = [ tupla[0] for tupla in eval_sol[:TASA_SUPERVIVENCIA]]
                nueva_generacion = padres + []

                # Creamos hijos con Blending, especificamente media ponderada.
                for i in range(HIJOS_GEN): # Uno para cada solucion faltante en la nueva poblacion.
                    w1 = random.random()
                    w2 = random.random()
                    x1 = random.choice(padres)
                    x2 = random.choice(padres)
                    child = ( (w1*x1 + w2*x2) / (w1+w2) )

                    # Mutaciones.
                    if random.random() <= PROB_MUTACION: child = child + random.gauss(0, VARIABILIDAD_MUTACION)

                    # Nos aseguramos de que el valor este dentro de nuestros limites
                    child = max(-10, min(10, child))
                    nueva_generacion.append(child)

                # La población de la SIGUIENTE iteración es la que acabamos de crear
                soluciones = nueva_generacion

            # Volvemos a evaluar y ordenar la población final para estar seguros
            eval_final = []
            for tupla in soluciones:
                eval_final.append((tupla, fitness_a(tupla)))
            eval_final.sort(key=lambda tupla: tupla[1], reverse=True)

            print("\n--- Proceso finalizado ---")
    
            # Requerimiento: Mostrar parámetros finales y población 
            print("\n--- Parámetros de la simulación ---")
            print(f"Tamaño Población: {TAMAÑO}")
            print(f"Umbral Diferencia: {UMBRAL_DIFERENCIA}")
            print(f"Variabilidad de Mutación: {VARIABILIDAD_MUTACION}")
            print(f"Generaciones transcurridas: {generacion_parada} / {GENERACIONES_MAXIMAS}")
            print(f"[FUNCIÓN] {func_text[funcion-1]}")
            print("\n--- Población Final (ordenada) ---")
            for i, (x, fitness) in enumerate(eval_final):
                print(f"{i+1}. x = {x:.8f} | y = {fitness:.8f}")
                
            # Devolvemos la población final por si se quiere guardar
            return soluciones
        
        case 2: # z = 20 + x - 10Cos(2pi*x) + y - 10Cos(2pi*y)
            best_fitness_anterior = -math.inf 
            generacion_parada = 0 # Para el reporte final
            for gen in range(GENERACIONES_MAXIMAS):
                generacion_parada += 1
                # Evaluamos la población.
                eval_sol = []
                for tupla in soluciones: eval_sol.append( (tupla, fitness_b(tupla[0], tupla[1])) )

                # Ordenamos de mayor a menor
                eval_sol.sort(key= lambda x: x[1], reverse=True)

                # Info:
                fitness_value = [tupla[1] for tupla in eval_sol]
                max_fitness = fitness_value[0]   # El primero (ya está ordenado)
                min_fitness = fitness_value[-1]  # El último (ya está ordenado)
                median_fitness = statistics.median(fitness_value)
                print(f"Generación: {gen+1} | Máximo: {max_fitness:.6f} | Mediana: {median_fitness:.6f} | Mínimo: {min_fitness:.6f}")

                #Chequeamos el umbral de diferencia
                diferencia = max_fitness - best_fitness_anterior
                # Si la mejora es menor que el umbral, paramos
                if diferencia > UMBRAL_DIFERENCIA:
                    # ¡Hubo mejora significativa!
                    best_fitness_anterior = max_fitness
                    generaciones_sin_mejora = 0         # Reseteamos la paciencia
                else:
                    # No hubo mejora significativa
                    generaciones_sin_mejora += 1        # Incrementamos la paciencia
                
                # Comprobamos si se nos acabó la paciencia
                if generaciones_sin_mejora >= PACIENCIA:
                    print(f"--- Convergencia por estancamiento en Gen {gen+1} ---")
                    print(f"No hubo mejora > {UMBRAL_DIFERENCIA} en {PACIENCIA} generaciones. Deteniendo.")
                    break # Salir del bucle 'for gen'                

                # Seleccionamos padres con elitismo.
                padres = [ tupla[0] for tupla in eval_sol[:TASA_SUPERVIVENCIA]]
                nueva_generacion = padres + []

                # Creamos hijos con Blending, especificamente media ponderada.
                for i in range(HIJOS_GEN): # Uno para cada solucion faltante en la nueva poblacion.
                    # Tuplas padres
                    padre1 = random.choice(padres) # Ej: (5.2, -3.1)
                    padre2 = random.choice(padres) # Ej: (1.0, 8.4)
                    
                    x1, y1 = padre1
                    x2, y2 = padre2

                    # Blending en x.
                    w1_x = random.random()
                    w2_x = random.random()
                    hx = ((w1_x * x1 + w2_x * x2) / (w1_x + w2_x))
                    
                    # Blending en y.
                    w1_y = random.random()
                    w2_y = random.random()
                    hy = ((w1_y * y1 + w2_y * y2) / (w1_y + w2_y))
                    
                    child = (hx, hy)
                    
                    # Mutamos en cada dimensión.
                    
                    mutated_hx, mutated_hy = hx, hy # Empezamos con los valores del hijo
                    
                    # Mutar x?
                    if random.random() <= PROB_MUTACION: mutated_hx = hx + random.gauss(0, VARIABILIDAD_MUTACION)
                    
                    # Mutar y?
                    if random.random() <= PROB_MUTACION: mutated_hy = hy + random.gauss(0, VARIABILIDAD_MUTACION)
                        
                    # Aplicamos los límites a ambas dimensiones
                    mutated_hx = max(-10, min(10, mutated_hx))
                    mutated_hy = max(-10, min(10, mutated_hy))
                    
                    child = (mutated_hx, mutated_hy)

                    nueva_generacion.append(child)

                # La población de la SIGUIENTE iteración es la que acabamos de crear
                soluciones = nueva_generacion

            # Volvemos a evaluar y ordenar la población final para estar seguros
            eval_final = []
            for tupla in soluciones:
                eval_final.append( (tupla, fitness_b(tupla[0], tupla[1])) )
            eval_final.sort(key=lambda tupla: tupla[1], reverse=True)

            print("\n--- Proceso finalizado ---")
    
            # Requerimiento: Mostrar parámetros finales y población 
            print("\n--- Parámetros de la simulación ---")
            print(f"Tamaño Población: {TAMAÑO}")
            print(f"Umbral Diferencia: {UMBRAL_DIFERENCIA}")
            print(f"Variabilidad de Mutación: {VARIABILIDAD_MUTACION}")
            print(f"Generaciones transcurridas: {generacion_parada} / {GENERACIONES_MAXIMAS}")
            print(f"[FUNCIÓN] {func_text[funcion-1]}")
            print("\n--- Población Final (ordenada) ---")
            for i, (x, fitness) in enumerate(eval_final):
                print(f"{i+1}. x = {x[0]:.8f} | y = {x[1]:.8f} | z = {fitness:.8f}")
                
            # Devolvemos la población final por si se quiere guardar
            return soluciones

# ✅
def testear_poblacion(poblacion, func):
    TAMAÑO = poblacion[0]
    UMBRAL_DIFERENCIA = poblacion[1]
    VARIABILIDAD_MUTACION = poblacion[3]
    soluciones = poblacion[4]
    func_text = ["y = ((sinx)^2) / x", "z = 20 + x - 10Cos(2pi*x) + y - 10Cos(2pi*y)"]
    match func:
        case 1:
            eval_sol = []
            for x in soluciones: eval_sol.append( (x, fitness_a(x)) )
            eval_sol.sort(key= lambda x: x[1], reverse=True)

            print("\n--- Parámetros de la simulación ---")
            print(f"Tamaño Población: {TAMAÑO}")
            print(f"Umbral Diferencia: {UMBRAL_DIFERENCIA}")
            print(f"Variabilidad de Mutación: {VARIABILIDAD_MUTACION}")
            print(f"[FUNCIÓN] {func_text[func-1]}")
            print("\n--- Población (ordenada) ---")
            for i, (x, fitness) in enumerate(eval_sol):
                print(f"{i+1}. x = {x:.8f} | y = {fitness:.8f}")
        case 2:
            eval_sol = []
            for tupla in soluciones: eval_sol.append( (tupla, fitness_b(tupla[0], tupla[1])) )
            eval_sol.sort(key= lambda x: x[1], reverse=True)

            print("\n--- Parámetros de la simulación ---")
            print(f"Tamaño Población: {TAMAÑO}")
            print(f"Umbral Diferencia: {UMBRAL_DIFERENCIA}")
            print(f"Variabilidad de Mutación: {VARIABILIDAD_MUTACION}")
            print(f"[FUNCIÓN] {func_text[func-1]}")
            print("\n--- Población (ordenada) ---")
            for i, (x, fitness) in enumerate(eval_sol):
                print(f"{i+1}. x = {x[0]:.8f} | y = {x[1]:.8f} | z = {fitness:.8f}")

# ✅
def definir_poblacion(func):
    tamano = 10
    umbral_diferencia = 0.05
    max_generaciones = 500
    variabilidad_mutacion = 1.0 
    while True:
        q = "0"
        q = input_options(q, (1,2,3,4,5,9), f"Población actual: | Tamaño: ({tamano}) | Umbral de diferencia: ({umbral_diferencia}) | Maximo de generaciones: ({max_generaciones}) | Variabilidad de la mutación ({variabilidad_mutacion}) " \
            "\n(1)Finalizar creación de la población. \n(2)Modificar tamaño de la población. \n(3)Modificar umbral de diferencia." \
            "\n(4)Modificar número máximo de generaciones. \n(5)Modificar variabilidad de la mutación.  \n(9)Volver. \n->")
                
        match q:
            case 1:
                if func == 1: soluciones = [random.uniform(-10,10) for i in range(tamano)]
                else: soluciones = [(random.uniform(-10,10),random.uniform(-10,10)) for i in range(tamano)] # [(x,y), (x,y), ...]
                return [tamano, umbral_diferencia, max_generaciones, variabilidad_mutacion, soluciones, func, 0] 
            case 2:
                tamano = "0"
                tamano = input_natural_n(tamano, "Tamaño de la población -> ")
            case 3:
                umbral_diferencia = "0"
                umbral_diferencia = input_positive_f(umbral_diferencia, "Umbral de diferencia -> ")
            case 4:
                max_generaciones = "0"
                max_generaciones = input_natural_n(max_generaciones, "Número máximo de generaciones -> ")
            case 5:
                variabilidad_mutacion = "0"
                variabilidad_mutacion = input_positive_f(variabilidad_mutacion, "Variabilidad de la mutación -> ")
            case 9:
                return False

def main():
    while True:
    # Selección de Población.
        q1="0"
        q1 = input_options(q1, (1,2,9), "\n¿Qué desea hacer? \n(1)Crear una nueva población. \n(2)Cargar población de un archivo. (*.csv) \n(9)Salir \n->")
        match q1:
            case 1:
                q2 = "0"
                q2 = input_options(q2, (1,2,9), "\n¿Qué función desea optimizar?\n" \
                "(1)y = ((sinx)^2) / x ; [ Maximizar dado x ∈ [-10,10] ]\n" \
                "(2)z = 20 + x - 10Cos(2pi*x) + y - 10Cos(2pi*y) ; [ Maximizar con valores x,y ∈ [-10,10] ]\n" \
                "(9)Volver\n" \
                "-> ")
                if q2==9: continue
                poblacion = definir_poblacion(q2)
                # [0:Tamaño, 1:Umbral de diferencia, 2: Maximo de generaciones, 3: Variabilidad de mutación, 4: Soluciones, 5:Tipo de función, 6: Cantidad de veces optimizada]
                if not poblacion: continue
            case 2:
                poblacion = cargar_poblacion()
                if not poblacion: continue
            case 9:
                break
        
        constantes = [0.25, 0.05, 0.25] # [paciencia, probabilidad de mutacion, tasa de supervivencia]
        func_text = ["y = ((sinx)^2) / x", "z = 20 + x - 10Cos(2pi*x) + y - 10Cos(2pi*y)"]
        while True:
            q3="0"
            print(f"\n--- [FUNCIÓN] {func_text[poblacion[5]-1]} ---")
            print(f"--- [POBLACIÓN] Tamaño: ({poblacion[0]}) | Umbral de diferencia: ({poblacion[1]}) | Maximo de generaciones: ({poblacion[2]}) | Variabilidad de la mutación ({poblacion[3]}) | Se ha optimizado ({poblacion[6]}) veces ---")
            print(f"--- [CONSTANTES] Paciencia: ({constantes[0]}) (% de generaciones totales dentro del umbral antes de terminar) | Probabilidad de mutación ({constantes[1]}) | Tasa de superviviencia ({constantes[2]}) ---")
            q3 = input_options(q3, (1,2,3,4,9), "¿Qué desea hacer? \n(1)Ejecutar algoritmo genético. \n(2)Testear población actual. \n(3)Guardar población \n(4)Modificar constantes \n(9)Volver \n->")
            match q3:
                case 1:
                    poblacion[4] = algoritmo_evolutivo(poblacion, constantes, poblacion[5])
                    poblacion[6] = poblacion[6]+1
                case 2:
                    testear_poblacion(poblacion, poblacion[5])
                case 3:
                    guardar_poblacion(poblacion)
                case 4:
                    constantes[0], constantes[1], constantes[2] = modificar_constantes()
                case 9:
                    break

        

        
main()