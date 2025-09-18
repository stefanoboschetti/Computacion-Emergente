import matplotlib.pyplot as plt
import csv, math, random

def read_csv_file(name,x,y): 
    try:
        with open(name) as csvfile:
            data = csv.reader(csvfile)
            for index, row in enumerate(data):
                dimensiones = len(row) - 1
                if index == 0 : 
                    for i in range(dimensiones): x.append([])
                    continue # CAMBIAR 
                for j in range(dimensiones): x[j].append(float(row[j]))
                y.append(float(row[dimensiones]))
        return True
    except:
        print("¡Error! Nombre de archivo invalido, el archivo debe estar en la misma carpeta.")
        return False

def suma(data,pesos,sesgo):
    temp_suma = []
    if not len(data[0]): return
    for value in range(len(data[0])):
        suma = 0
        for dimension in range(len(data)):
            suma += data[dimension][value]*pesos[dimension] + sesgo
        temp_suma.append(suma)
    return temp_suma

def activation_function(tipo, data):
    temp_predicciones = []
    match tipo:
        case 1:
            for value in data:
                temp_predicciones.append(1) if value > 0 else temp_predicciones.append(0) 
        case 2:
            for value in data:
                temp_predicciones.append(1/(1+math.exp(-value)))
        case 3:
            for value in data:
                temp_predicciones.append(1/(1+math.exp(-value)))
        
    return temp_predicciones

def verify_colors(predicted, real, tipo, error):
    temp_colores_coincidencias = []
    temp_colores_esperados = []
    temp_colores_prediccion = []
    for index, value in enumerate(predicted):
        if tipo == 1:
            temp_colores_coincidencias.append((0,0.6,0,1)) if value == round(real[index]) else temp_colores_coincidencias.append( (1, 0, 0, 0.5) )
        elif tipo == 2:
            if round(value) == round(real[index]): temp_colores_coincidencias.append((0,0.6,0,1))
            else: temp_colores_coincidencias.append( (1, 0, 0, 0.5) )
        elif tipo == 3:
            if abs(value-real[index]) < error: temp_colores_coincidencias.append((0,0.6,0,1))
            else: temp_colores_coincidencias.append( (1, 0, 0, 0.5) )
        temp_colores_esperados.append( (0, 0.6, 0, real[index]) ) if real[index] > 0 else temp_colores_esperados.append( (1, 0, 0, 0.5) )
        temp_colores_prediccion.append( (0, 0.6, 0, value) ) if value > 0 else temp_colores_prediccion.append ( (1, 0, 0, 0.5) )
    
    return temp_colores_coincidencias, temp_colores_esperados, temp_colores_prediccion

def main():
    x = []
    y = []
    valores_suma = []
    predicciones = []
    colores_esperados = []
    colores_prediccion = []
    colores_coincidencias = []
    pesos = []
    sesgo =  0
    bad_file = True
    error = -1  

    while bad_file:
        file_option = 0
        while int(file_option) not in (1,2,3):
            file_option = input('Seleccione un archivo \n(1)fuzzy_separables.csv \n(2)no_separables.csv \n(3)Otro \n-> ')
        
        match int(file_option):
            case 1: file_option = 'fuzzy_separables.csv'
            case 2: file_option = 'no_separables.csv'
            case 3: file_option = input('Porfavor introduce el nombre del archivo ".csv" (e.g. "datos.csv")-> ')
        bad_file = not read_csv_file(file_option,x,y)

    file_name = 0
    while int(file_name) not in (1,2,3):
        file_name = int(input('Indica una función de activación: \n(1)Escalonada \n(2)Sigmoidal (Redondeo)\n(3)Sigmoidal (Error) \n-> '))

    while True:
        if file_name == 3:
            error = input('Indique un valor de error [0,1)\n-> ')
        if error == "": error = 0.15
        else: error = float(error)

        print(f"Error: {error}\n")
        ask_pesos = input(f"Introduzca los {len(x)} pesos separados por comas. (e.g. 1.5,-0.8) \nDEJAR EN BLANCO PARA ALEATORIO [-2,2]\n-> ")
        if not ask_pesos: pesos = [float(random.uniform(-2,2)) for x in range(len(x))]
        else: pesos = [float(x) for x in ask_pesos.split(",")]
        print(F"Pesos: {pesos}")
        print(F"\n-----------------")
        sesgo = input(f"\nIntroduza el sesgo o bias. (e.g. 0.22)\nDEJAR EN BLANCO PARA ALEATORIO [-0.5,0.5]\n-> ")
        if sesgo == "" : 
            sesgo = random.uniform(-0.5,0.5)
            print(F"Sesgo: {sesgo}")
        else: sesgo = float(sesgo)

        valores_suma = suma(x,pesos,sesgo)

        predicciones = activation_function(file_name, valores_suma)

        colores_coincidencias, colores_esperados, colores_prediccion = verify_colors(predicciones, y, file_name,error)

        print("")
        if file_name == 3: print(f"Coincidencias con error de {error}")
        print("Cierre TODAS las ventanas para continuar.\n")

        plt.figure("Valor Esperado")
        plt.scatter(x[0],x[1], c=colores_esperados, label="1")
        plt.xlabel("Rojo: Menor o igual a 0, Verde: Mayor intensidad al acercarse de 1")
        plt.get_current_fig_manager().window.geometry("+0+400")

        plt.figure("Valor Predicho")
        plt.scatter(x[0],x[1], c=colores_prediccion, label="2")
        plt.xlabel("Rojo: Menor o igual a 0, Verde: Mayor intensidad al acercarse de 1")
        plt.get_current_fig_manager().window.geometry("+600+400")

        plt.figure("Coincidencias")
        plt.scatter(x[0],x[1], c=colores_coincidencias, label="2")
        plt.xlabel("Rojo: No coinciden, Verde: Coinciden")
        plt.get_current_fig_manager().window.geometry("+1200+400")
        plt.show()


        end_program = 0
        while int(end_program) not in (1,2):
            end_program = input("Desea repetir con otros valores?\n(1)Sí\n(2)No\n-> ")
        if end_program=="2": break

main()