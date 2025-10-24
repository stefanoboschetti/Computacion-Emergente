import csv, math, random

# Validar que los inputs de texto sean correctos c:
def inputv(var, valid_values, text):
    while(not str(var).isnumeric() or int(var) not in valid_values):
        var = input(text)
    return int(var)

# ✅
def sigmoidal(value):
    return 1/(1+math.exp(-value))

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

def backpropagation():
    pass

# ✅
def feedforward(red, sizes):
    x = []
    y = []
    while(True):
        q3 = "0"
        try:
            q3 = inputv(q3, (1,2,3,4,5,9),f"Selecciona un archivo para realizar las pruebas \n(1)2-d_2-class_test \n(2)3-d_4-class_test \n(3)Otro \n(9)Volver \n-> ")
            match(q3):
                case 1:
                    file_name = "2-d_2-class_test"
                case 2:
                    file_name = "3-d_4-class_test"
                case 3:
                    file_name = input(f"Ingrese el nombre del archivo. (No incluir '.csv') \nPor la forma de la red neuronal:\nLos datos deben tener {sizes[0]}d de entrada y {sizes[1]}d de salida.  \n->")
                case 9:
                    return

            x,y = read_csv(file_name)
            dimensiones_entrada = len(x)
            if dimensiones_entrada != sizes[0] or 1 != sizes[1]:
                print(f"¡Error! Las dimensiones de entrada/salida no son compatibles con la red neuronal de {sizes[0]}d de entrada y {sizes[1]}d de salida.\n")
            else:
                break
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")
    
    # Ahora si empezamos con el feedforward.
    # red [capa] [conexiones] [conexion] [peso, sesgo]
    resultados = []
    # Evaluamos la data de testeo.
    for j, valor in enumerate(x[0]):
        entradas = []
        # Populamos las entradas iniciales.
        for i in range(dimensiones_entrada): 
            entradas.append(x[i][j])
        # Evaluamos las entradas.
        for capa in red:
            nuevas_entradas = []
            for conexiones in capa:
                sumatoria = 0
                count_entradas = 0
                for i, conexion in enumerate(conexiones):
                    sumatoria = sumatoria + (entradas[count_entradas] * conexion)
                    count_entradas = count_entradas + 1
                    if count_entradas > len(entradas)-1: count_entradas = 0
                nuevas_entradas.append(sigmoidal(sumatoria))
            entradas = nuevas_entradas
        resultados.append(nuevas_entradas[0]) # AQUI TENEMOS EL RESULTADO
    for i in resultados:
        print(i)
    return

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
        if q2 == 1:    backpropagation()
        elif q2 == 2:  feedforward(red_neuronal, sizes)
        elif q2 == 3:  save_network(red_neuronal, sizes)
        elif q2 == 9:  return

main()