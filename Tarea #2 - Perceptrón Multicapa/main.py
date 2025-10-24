import csv, math, random

# Validar que los inputs de texto sean correctos c:
def inputv(var, valid_values, text):
    while(not var.isnumeric() or int(var) not in valid_values):
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
                for j in range(dimensiones): x[j].append(float(row[j]))
                y.append(float(row[dimensiones]))
        return x,y

# ✅
def create_perceptron():
    neuronas_entrada = int(input("Ingrese el número de neuronas de entrada. \n->"))
    neuronas_salida = int(input("Ingrese el número de neuronas de salida. \n->"))
    capas_ocultas = int(input("Ingrese el número de capas ocultas. \n->"))
    neuronas_capa = int(input("Ingrese el número de neuronas por capa. \n->"))

    # Pesos para entradas.
    entrada = [ [[round(random.uniform(-5,5),1), round(random.uniform(-5,5),1)] for j in range(neuronas_capa)] for i in range(neuronas_entrada) ]
    # Pesos para capas ocultas.
    red = [ [ [ [round(random.uniform(-5,5),1), round(random.uniform(-5,5),1)] for k in range(neuronas_capa) ] for j in range (neuronas_capa)] for i in range(capas_ocultas-1) ]
    # Pesos de capa final que se comunica con la salida.
    red.append([ [ [round(random.uniform(-5,5),1), round(random.uniform(-5,5),1)] for k in range(neuronas_salida) ] for j in range (neuronas_capa)])
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
                        entrada = [ [[0,0] for j in range(sizes[3])] for i in range(sizes[0]) ]
                        # Pesos para capas ocultas.
                        red = [ [ [ [0,0] for k in range(sizes[3]) ] for j in range (sizes[3])] for i in range(sizes[2]-1) ]
                        # Pesos de capa final que se comunica con la salida.
                        red.append([ [ [0,0] for k in range(sizes[1]) ] for j in range (sizes[3])])
                        # Concatenamos la entrada al inicio.
                        red = [entrada] + red
                    else:
                        k = 0
                        l = 0
                        for i, neurona in enumerate(red[index-1]):
                            for j, conexion in enumerate(neurona):
                                red[index-1][i][j][0] = float(row[k])
                                red[index-1][i][j][1] = float(row[k+1])
                                k = k+2
            print(red)
            return red, sizes                 
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")

def backpropagation():
    pass

def feedforward(red, sizes):
    x = []
    y = []
    while(True):
        # file_name = input(f"Ingrese el nombre del archivo. (No incluir '.csv') \nPor la forma de la red neuronal:\nLos datos deben tener {sizes[0]}d de entrada y {sizes[1]}d de salida.  \n->")
        file_name = "2-d_2-class_test"
        x,y = read_csv(file_name)
        dimensiones_entrada = len(x)
        if dimensiones_entrada != sizes[0] or 1 != sizes[1]:
            print(f"¡Error! Las dimensiones de entrada/salida no son compatibles con la red neuronal de {sizes[0]}d de entrada y {sizes[1]}d de salida.\n")
            continue
        
        # Ahora si empezamos con el feedforward.
       
        
        for capa in red:
            for j, valor in enumerate(x[0]):
                for c in range(len(capa[0][0])*2):
                    for dimension in range(dimensiones_entrada):
                            print(capa[dimension][c])

        return


# ✅
def save_network(red, sizes):
    # Transformamos la red en una estructura mas conveniente para el guardado.
    data_rows = [sizes]
    for index, capa in enumerate(red):
        data_rows.append([])
        for neurona in capa:
            for conexion in neurona:
                data_rows[index+1].append(conexion[0]) # Peso
                data_rows[index+1].append(conexion[1]) # Sesgo

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
    
    q2  = inputv(q2, (1,2,3,9), "¿Qué desea hacer ahora? \n(1)Entrenar la red con retropropagación. \n(2)Probar la red con alimentación hacia delante. \n(3)Guardar la red. \n(9)Salir \n->")
    if q2 == 1:    backpropagation()
    elif q2 == 2:  feedforward(red_neuronal, sizes)
    elif q2 == 3:  save_network(red_neuronal, sizes)
    elif q2 == 9:  return

main()