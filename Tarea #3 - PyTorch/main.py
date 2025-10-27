import torch
import csv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.transforms import ToTensor, Lambda
from torch import nn

# ✅
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    error_epoca = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss, current = loss.item(), batch * batch_size + len(X)
        error_epoca += loss
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    error_epoca /= len(dataloader)
    return error_epoca

# ✅
def test(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    preds = torch.tensor([])
    ys = torch.tensor([])
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            preds = torch.concat((preds, pred.argmax(1)), dim=0)
            ys = torch.concat((ys, y), dim=0)

    test_loss /= num_batches
    correct /= size
    print(f"\n[TEST] Avg loss: {test_loss:>8f}")
    return preds, ys

# ✅
def graficar_error_epoca(epochs, errores):
    plt.figure("Perdida por época")
    plt.plot(range(1, epochs+1), errores, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Perdida (loss)")
    plt.title("Perdida (loss) de la red durante el entrenamiento")
    plt.grid(True)
    plt.show()

# ✅
def mostrar_metricas(conf_matrix):
    # Por clase
    TP = conf_matrix.diag()
    FP = conf_matrix.sum(0) - TP
    FN = conf_matrix.sum(1) - TP

    precision_per_class = TP / (TP + FP + 1e-8)
    recall_per_class    = TP / (TP + FN + 1e-8)

    # Promedio macro
    precision_macro = precision_per_class.mean()
    recall_macro    = recall_per_class.mean()

    # Accuracy general
    accuracy = TP.sum() / conf_matrix.sum()

    print("\n--- Métricas ---")
    print(f"Accuracy : {100*accuracy:.2f}%")
    print(f"Precision: {100*precision_macro:.2f}%")
    print(f"Recall   : {100*recall_macro:.2f}%")

# ✅
def graficar_matriz_confusion(preds, ys):
    num_classes = 10
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Llenar la matriz de confusión
    for t, p in zip(ys, preds):
        conf_matrix[t.long(), p.long()] += 1
    
    plt.figure(figsize=(8,6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de Confusión - FashionMNIST')
    plt.colorbar()

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(num_classes), class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j].item(),
                    ha='center', va='center',
                    color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')

    plt.tight_layout()
    mostrar_metricas(conf_matrix)
    plt.show()

# ✅
def get_FasionMNIST_dataloaders(batch_size):
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader

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
def entrenar_modelo(model, batch_size, epochs, learning_rate):
    train_dataloader, test_dataloader = get_FasionMNIST_dataloaders(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    errores = []
    for t in range(epochs):
        print(f"\n[Epoch {t+1}]\n-------------------------------")
        errores.append( train_loop(train_dataloader, model, loss_fn, optimizer, batch_size) )
    print("Done!")
    return errores

# ✅
def testear_modelo(model, batch_size, epochs, learning_rate):
    train_dataloader, test_dataloader = get_FasionMNIST_dataloaders(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    preds, ys = test(test_dataloader, model, loss_fn)
    return preds, ys

# ✅
def modificar_hiperparametros():
    batch_size = epochs = learning_rate = "0"
    batch_size = input_natural_n(batch_size, "Batch size (e.g. 64) -> ")
    epochs = input_natural_n(epochs, "Número de epocas (e.g. 10) -> ")
    learning_rate = input_positive_f(learning_rate, "Learning rate (e.g. 0.05)-> ")
    return batch_size, epochs, learning_rate

# ✅
def cargar_modelo():
    while(True):
        try:
            file_name = input("Ingrese el nombre del archivo. (No incluir '.pth')\n -> ")
            state_dict = torch.load(f'./models/{file_name}.pth', weights_only=False)
            with open(f"./models/{file_name}.csv") as file:
                data = csv.reader(file)
                for index, row in enumerate(data):
                    model = crear_modelo(int(row[0]), int(row[1]), int(row[2]))
                    model.load_state_dict(state_dict)  
                    return model
        except:
            print("¡Error! Nombre de archivo invalido o archivo incompatible.")
            return False

# ✅
def guardar_modelo(model):
    while(True):
        file_name = input("Ingrese el nombre del archivo. (No incluir '.pth')\n -> ")
        torch.save(model.state_dict(), f'./models/{file_name}.pth')   
        data_rows = [model.tipo, model.capas, model.neuronas]
        with open(f"./models/{file_name}.csv", mode='w', newline='') as file:
            csv.writer(file).writerow(data_rows)
        return 

# ➖
def generar_argumentos(tipo_red, capas_ocultas, neuronas_capa):
    arguments = [[],[],[]] # model, features, classifier
    match tipo_red:

        case 0:
            #Entrada a capas ocultas.
            arguments[0].append( nn.Linear(28*28, neuronas_capa) )
            arguments[0].append( nn.ReLU() )
            #Capas ocultas.
            for capa in range(capas_ocultas-1):
                arguments[0].append( nn.Linear(neuronas_capa, neuronas_capa) )
                arguments[0].append( nn.ReLU() )
            # Capas ocultas a salida.
            arguments[0].append( nn.Linear(neuronas_capa, 10) )

        case 1:
            # #Entrada a capas ocultas.
            # arguments.append( nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) )
            # arguments.append( nn.BatchNorm2d(32) )
            # arguments.append( nn.ReLU() )
            # #Capas ocultas.
            # for capa in range(capas_ocultas-1):
            #     arguments.append( nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) )
            # arguments.append( nn.BatchNorm2d(32) )
            # arguments.append( nn.ReLU() )
            # # Capas ocultas a salida.
            # arguments.append(nn.Linear(neuronas_capa, 10))
            pass

    return arguments

# ➖
def crear_modelo(tipo_red, capas_ocultas, neuronas_capa):
    class NeuralNetwork(nn.Module):
        tipo = tipo_red
        capas = capas_ocultas
        neuronas = neuronas_capa
        def __init__(self):
            model_args, features_args, classifier_args = generar_argumentos(tipo_red, capas_ocultas, neuronas_capa)
            super().__init__()
            self.flatten = nn.Flatten()
            self.model = nn.Sequential(*model_args)
            self.features = nn.Sequential(*features_args)
            self.classifier = nn.Sequential(*classifier_args)

        def forward(self, x):
            match tipo_red:
                case 0:
                    x = self.flatten(x)
                    logits = self.model(x)
                    return logits

                case 1:
                    pass
        
    return NeuralNetwork()

# ✅
def crear_red():
    capas_ocultas = 2
    neuronas_capa = 512
    tipo_red = 0 # 0:Rectangular, Convolucional
    while True:
        redes = (f"Rectangular | ({neuronas_capa}) neuronas por capa", "Convolucional")

        q1 = "0"
        q1 = input_options(q1, (1,2,3,9), f"\nRed actual: | {redes[tipo_red]} | ({capas_ocultas}) capas ocultas | \n(1)Finalizar creación de la red. \n(2)Modificar número de capas ocultas. \n(3)Modificar tipo de red (Cambiar nodos por capa en red rectangular). \n(9)Volver. \n->")
        
        match q1:
            case 1:
                return crear_modelo(tipo_red, capas_ocultas, neuronas_capa) #Aquí pasamos una red.
            case 2:
                capas_ocultas = "0"
                capas_ocultas = input_natural_n(capas_ocultas, "Número de capas ocultas -> ")
            case 3:
                tipo_red = "-1"
                tipo_red = input_options(tipo_red, (1,2), f"(1)Rectangular \n(2)Convolucional \n-> ") - 1
                if tipo_red == 0:
                    neuronas_capa = "0"
                    neuronas_capa = input_natural_n(neuronas_capa, "Número de neuronas por capa -> ")
            case 9:
                return False


def main():
    while True:
    # Selección de Red Neuronal.
        q1="0"
        q1 = input_options(q1, (1,2,9), "\n¿Qué desea hacer? \n(1)Crear una nueva red. \n(2)Cargar red de un archivo. (*.pth) \n(9)Salir. \n->")
        match q1:
            case 1:
                model = crear_red()
                if not model: continue
            case 2:
                model = cargar_modelo()
                if not model: continue
            case 9:
                break

    # Realizar acción en Red Neuronal.
        batch_size = 64
        epochs = 10
        learning_rate = 0.001
        while True:
            q2="0"
            redes = (f"Rectangular | ({model.neuronas}) neuronas por capa", "Convolucional")
            print(f"\n--- [RED] | {redes[model.tipo]} | ({model.capas}) capas ocultas | ---")
            print(f"--- [Hiperparametros] | Epocas: ({epochs}) | Batch Size: ({batch_size}) | Learning Rate: ({learning_rate}) | ---")
            q2 = input_options(q2, (1,2,3,4,9), "¿Qué desea hacer? \n(1)Entrenar la red. \n(2)Testear la red. \n(3)Guardar la red. \n(4)Modificar Hiperparametros. \n(9)Volver a seleccionar una Red. \n->")
            match q2:
                case 1:
                    errores = entrenar_modelo(model, batch_size, epochs, learning_rate)
                    graficar_error_epoca(epochs, errores)
                case 2:
                    preds, ys = testear_modelo(model, batch_size, epochs, learning_rate)
                    graficar_matriz_confusion(preds, ys)
                case 3:
                    guardar_modelo(model)
                case 4:
                    batch_size, epochs, learning_rate = modificar_hiperparametros()
                case 9:
                    break

main()