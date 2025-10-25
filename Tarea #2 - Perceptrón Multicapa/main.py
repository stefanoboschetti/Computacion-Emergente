import csv, math, random
import matplotlib.pyplot as plt
import numpy as np

# Validar que los inputs de texto sean correctos c:
def inputv(var, valid_values, text):
    while(not str(var).isnumeric() or int(var) not in valid_values):
        var = input(text)
    return int(var)

# ✅
def sigmoidal(value):
    return 1/(1+math.exp(-value))

# ✅
def derivada_sigmoidal(value):
    return sigmoidal(value)*(1-sigmoidal(value))

# ✅
def read_csv(name):
        x = []
        y = []
        with open(f"{name}.csv") as csvfile:
            data = csv.reader(csvfile)
            for index, row in enumerate(data):
                dimensiones = len(row) - 1
                if index == 0 : #Ignoramos la primera columna con nombres.
                    for i in range(dimensiones): x.append([])
                    continue 
                for j in range(dimensiones):
                    if not row[j]: continue
                    x[j].append(float(row[j]))
                if not row[dimensiones]: continue
                y.append(float(row[dimensiones]))
        return x,y

# ✅
def create_perceptron():
    neuronas_entrada = int(input("Ingrese el número de neuronas de entrada. \n->"))
    neuronas_salida = int(input("Ingrese el número de neuronas de salida. \n->"))
    capas_ocultas = int(input("Ingrese el número de capas ocultas. \n->"))
    neuronas_capa = int(input("Ingrese el número de neuronas por capa. \n->"))

    # Pesos para entradas.
    entrada = [ [ round(random.uniform(-5,5),1) for j in range(neuronas_capa)] for i in range(neuronas_entrada) ]
    # Pesos para capas ocultas.
    red = [ [ [ round(random.uniform(-5,5),1) for k in range(neuronas_capa) ] for j in range (neuronas_capa)] for i in range(capas_ocultas-1) ]
    # Pesos de capa final que se comunica con la salida.
    red.append([ [ round(random.uniform(-5,5),1) for k in range(neuronas_capa) ] for j in range (neuronas_salida)])
    # Concatenamos la entrada al inicio.
    red = [entrada] + red
    print(red)
    return red, [neuronas_entrada, neuronas_salida, capas_ocultas, neuronas_capa]

# ✅
def load_perceptron():
    sizes = []
    red = []
    while(True):
        file_name = input("Ingrese el nombre del archivo. (No incluir '.csv')\n ->")
        try:
            with open(f"{file_name}.csv") as file:
                data = csv.reader(file)
                for index, row in enumerate(data):
                    if index == 0: 
                        sizes = [int(i) for i in row]
                        # Pesos para entradas.
                        entrada = [ [ 0 for j in range(sizes[3])] for i in range(sizes[0]) ]
                        # Pesos para capas ocultas.
                        red = [ [ [ 0 for k in range(sizes[3]) ] for j in range (sizes[3])] for i in range(sizes[2]-1) ]
                        # Pesos de capa final que se comunica con la salida.
                        red.append([ [ 0 for k in range(sizes[3]) ] for j in range (sizes[1])])
                        # Concatenamos la entrada al inicio.
                        red = [entrada] + red
                    else:
                        k = 0
                        for i, neurona in enumerate(red[index-1]):
                            for j, conexion in enumerate(neurona):
                                red[index-1][i][j] = float(row[k])
                                k = k+1
            print(red)
            return red, sizes                 
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")

# ✅
def do_backpropagation(red, sizes):
    while True:
        q4 = "0"
        try:
            q4 = inputv(q4, (1,2,3,4,5,9), "Selecciona un archivo para realizar las pruebas \n(1)2-d_2-class_train \n(2)3-d_4-class_train \n(3)Otro \n(9)Volver \n-> ")
            match q4:
                case 1: file_name = "2-d_2-class_train"
                case 2: file_name = "3-d_4-class_train"
                case 3: file_name = input(f"Ingrese el nombre del archivo. (No incluir '.csv') \nPor la forma de la red neuronal:\nLos datos deben tener {sizes[0]}d de entrada y {sizes[1]}d de salida.  \n->")
                case 9: return

            x, y = read_csv(file_name)
            if len(x) != sizes[0] or sizes[1] != 1:
                print(f"¡Error! Las dimensiones de entrada/salida no son compatibles con la red neuronal de {sizes[0]}d de entrada y {sizes[1]}d de salida.\n")
            else: break
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")
    
    q4 = ""
    while not q4.isnumeric(): q4 = input("Número de épocas a realizar?\n ->")
    q4 = int(q4)

    errores = []  # Para guardar error por época
    alpha = float(input("Learning Rate (e.g. 0.05)-> "))
    for epoch in range(q4):
        backpropagation(red, file_name, alpha)
        # Calcular error cuadrático medio
        resultados, _ = feedforward(red, x)
        y_norm, _ = normalize_classes(y)
        error_epoca = 0
        for i in range(len(x[0])):
            error_epoca += (resultados[i][-1][0] - y_norm[i])**2
        error_epoca /= len(x[0])
        errores.append(error_epoca)
        print(f"Época {epoch+1} - Error: {error_epoca:.4f}")

    # Mostrar gráfico de error por época
    plt.figure("Error por época")
    plt.plot(range(1, q4+1), errores, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Error cuadrático medio")
    plt.title("Error de la red durante el entrenamiento")
    plt.grid(True)
    plt.show()

    # Mostrar scatter con coincidencias
    show_scatter(resultados, x, y)
    resultados, _ = feedforward(red, x)

# ✅
def classify(value,y):
    y_norm, mapping = normalize_classes(y)
    return min(mapping.keys(), key=lambda c: abs(mapping[c] - value))

# ✅
def normalize_classes(y):
    classes = sorted(list(set(y)))
    n = len(classes)
    mapping = {c: i/(n-1) if n > 1 else 0.5 for i, c in enumerate(classes)}
    return [mapping[val] for val in y], mapping

# ✅
def backpropagation(red, file_name, a):
    alpha = a
    x, y = read_csv(file_name)
    y_norm, mapping = normalize_classes(y)
    resultados, sumatorias = feedforward(red, x)

    for idx_dato in range(len(x[0])):
        deltas = [None] * len(red)
        capa_salida = len(red) - 1
        deltas[capa_salida] = []

        # ---- Calcular deltas de salida ----
        for j in range(len(red[capa_salida])):
            out = resultados[idx_dato][-1][j]
            net = sumatorias[idx_dato][-1][j]
            target = y_norm[idx_dato]
            deltas[capa_salida].append((out - target) * derivada_sigmoidal(net))

        # ---- Calcular deltas de capas ocultas ----
        for capa in range(len(red) - 2, -1, -1):
            deltas[capa] = []
            for j in range(len(red[capa])):
                suma = 0.0
                for k in range(len(red[capa + 1])):
                    if j < len(red[capa + 1][k]):
                        peso = red[capa + 1][k][j]
                        suma += deltas[capa + 1][k] * peso
                net = sumatorias[idx_dato][capa][j]
                deltas[capa].append(derivada_sigmoidal(net) * suma)

        # ---- Actualizar pesos ----
        entradas = [x[i][idx_dato] for i in range(len(x))]

        for capa in range(len(red)):
            for j in range(len(red[capa])):
                for k in range(len(red[capa][j])):
                    if capa == 0 and k < len(entradas):
                        entrada_valor = entradas[k]
                    elif capa > 0 and k < len(resultados[idx_dato][capa - 1]):
                        entrada_valor = resultados[idx_dato][capa - 1][k]
                    else:
                        entrada_valor = 0.0

                    red[capa][j][k] -= alpha * deltas[capa][j] * entrada_valor


def mostrar_matriz_confusion(y_real, y_pred):
    classes = sorted(list(set(y_real)))
    n_classes = len(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}

    # Crear matriz
    mat_conf = np.zeros((n_classes, n_classes))
    for i in range(len(y_real)):
        mat_conf[class_to_idx[y_real[i]], class_to_idx[y_pred[i]]] += 1

    plt.figure("Matriz de Confusión")
    plt.imshow(mat_conf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicho")
    plt.ylabel("Real")

    # Mostrar los números dentro de cada celda
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, int(mat_conf[i, j]), ha="center", va="center", color="red")

    plt.show()

# ✅
def do_feedforward(red, sizes):
    while True:
        q3 = "0"
        try:
            q3 = inputv(q3, (1,2,3,4,5,9), "Selecciona un archivo para realizar las pruebas \n(1)2-d_2-class_test \n(2)3-d_4-class_test \n(3)Otro \n(9)Volver \n-> ")
            match q3:
                case 1: file_name = "2-d_2-class_test"
                case 2: file_name = "3-d_4-class_test"
                case 3: file_name = input(f"Ingrese el nombre del archivo. (No incluir '.csv') \nPor la forma de la red neuronal:\nLos datos deben tener {sizes[0]}d de entrada y {sizes[1]}d de salida.  \n->")
                case 9: return

            x, y = read_csv(file_name)
            if len(x) != sizes[0] or sizes[1] != 1:
                print(f"¡Error! Las dimensiones de entrada/salida no son compatibles con la red neuronal de {sizes[0]}d de entrada y {sizes[1]}d de salida.\n")
            else: break
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")
    
    resultados, _ = feedforward(red, x)
    show_scatter(resultados, x, y)

    # ---------------- Matriz de confusión ----------------
    pred = [classify(resultados[i][-1][0], y) for i in range(len(y))]
    mostrar_matriz_confusion(y,pred)

# ✅
def feedforward(red, x):
    # red [capa] [conexiones] [conexion] [peso, sesgo]
    resultados = []
    sumatorias = []
    count_s = 0
    count_r = 0
    # Evaluamos la data de testeo.
    for j, valor in enumerate(x[0]):
        entradas = []
        resultados.append([])
        sumatorias.append([])
        # Populamos las entradas iniciales.
        for i in range(len(x)): 
            entradas.append(x[i][j])
        # Evaluamos las entradas.
        for capa in red:
            nuevas_entradas = []
            ins = []
            for conexiones in capa:
                sumatoria = 0
                count_entradas = 0
                for i, conexion in enumerate(conexiones):
                    sumatoria = sumatoria + (entradas[count_entradas] * conexion)
                    count_entradas = count_entradas + 1
                    if count_entradas > len(entradas)-1: count_entradas = 0
                nuevas_entradas.append(sigmoidal(sumatoria))
                ins.append(sumatoria)
            entradas = nuevas_entradas
            resultados[count_r].append(entradas)
            sumatorias[count_s].append(ins)
        count_r = count_r + 1
        count_s = count_s + 1
    return resultados, sumatorias

def show_scatter(resultados, x, y):
    colores = [
        ((1.0, 0.0, 0.0, 1.0), "Rojo"),
        ((0.0, 1.0, 0.0, 1.0), "Verde"),
        ((0.0, 0.0, 1.0, 1.0), "Azul"),
        ((1.0, 1.0, 0.0, 1.0), "Amarillo"),
        ((1.0, 0.65, 0.0, 1.0), "Naranja"),
        ((0.5, 0.0, 0.5, 1.0), "Púrpura"),
        ((0.0, 1.0, 1.0, 1.0), "Cian"),
        ((1.0, 0.0, 1.0, 1.0), "Magenta"),
        ((0.5, 0.5, 0.5, 1.0), "Gris"),
        ((0.0, 0.0, 0.0, 1.0), "Negro"),
        ((0.72, 0.53, 0.04, 1.0), "Oro"),
        ((0.55, 0.27, 0.07, 1.0), "Marrón"),
        ((0.25, 0.88, 0.82, 1.0), "Turquesa"),
        ((0.93, 0.51, 0.93, 1.0), "Orquídea"),
        ((0.86, 0.08, 0.24, 1.0), "Carmesí"),
        ((0.18, 0.55, 0.34, 1.0), "Verde bosque"),
        ((0.29, 0.0, 0.51, 1.0), "Índigo"),
        ((1.0, 0.75, 0.8, 1.0), "Rosa"),
        ((0.68, 0.85, 0.9, 1.0), "Celeste"),
        ((0.5, 0.0, 0.0, 1.0), "Granate")
    ]
    colores.reverse()
    classes = []
    texto = ""
    coincidencias = []
    scatter_colors = []
    
    for i, yi in enumerate(y):
        if yi not in classes: classes.append(yi)
        if yi == classify(resultados[i][-1][0], y): coincidencias.append(colores[-2][0])
        else: coincidencias.append(colores[-1][0])
    
    for i in y: 
        scatter_colors.append(colores[classes.index(i)][0])
    for j in classes: texto += f"Clase[{j}]: {colores[classes.index(j)][1]} | "

    if (len(x)>2): 
        plt.figure("Valor Esperado").add_subplot(projection='3d').scatter(x[0],x[1],x[2], c=scatter_colors, label="1")
    else:
        plt.figure("Valor Esperado")
        plt.scatter(x[0],x[1], c=scatter_colors, label="1")
    plt.xlabel(texto)
    plt.get_current_fig_manager().window.geometry("+0+400")

    if (len(x)>2): 
        plt.figure("Coincidencias").add_subplot(projection='3d').scatter(x[0],x[1],x[2], c=coincidencias, label="2")
    else: 
        plt.figure("Coincidencias")
        plt.scatter(x[0],x[1], c=coincidencias, label="2")
    plt.xlabel("Rojo: No coinciden, Verde: Coinciden")
    plt.get_current_fig_manager().window.geometry("+1200+400")
    plt.show()

# ✅
def save_network(red, sizes):
    # Transformamos la red en una estructura mas conveniente para el guardado.
    data_rows = [sizes]
    for index, capa in enumerate(red):
        data_rows.append([])
        for neurona in capa:
            for conexion in neurona:
                data_rows[index+1].append(conexion) # Peso

    file_name = input("Ingrese el nombre del archivo. (No incluir '.csv')\n ->")
    with open(f"{file_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_rows)

def main():
    sizes = [0,0,0,0] # Entrada, salida, capas, neuronasXcapa.
    red_neuronal = [] # Red neuronal completa en forma de pesos.
    q1=q2="0"         # Preguntas

    q1  = inputv(q1, (1,2,9), "¿Qué desea hacer? \n(1)Crear nuevo perceptrón multicapa. \n(2)Cargar de un archivo. \n(9)Salir \n->")
    if q1 == 1:    red_neuronal, sizes = create_perceptron()
    elif q1 == 2:  red_neuronal, sizes = load_perceptron()
    elif q1 == 9:  return


    while(True):
        q2="0"
        q2  = inputv(q2, (1,2,3,9), "¿Qué desea hacer ahora? \n(1)Entrenar la red con retropropagación. \n(2)Probar la red con alimentación hacia delante. \n(3)Guardar la red. \n(9)Salir \n->")
        if q2 == 1:    do_backpropagation(red_neuronal, sizes)
        elif q2 == 2:  do_feedforward(red_neuronal, sizes)
        elif q2 == 3:  save_network(red_neuronal, sizes)
        elif q2 == 9:  return

main()