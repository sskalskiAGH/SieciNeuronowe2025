import pandas as pd
import numpy as np

# Funkcja aktywacji: Sigmoid - jest skuteczna jako funkcja aktywacji dla problemow klasyfikacji zmiennych binarnych, przyjmuje wartości między 0 a 1 
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Inicjalizacja wag sieci
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1 # macierz o wymiarach: liczba neuronow w warstwie wejsciowej X liczba neuronow w warstwie ukrytej
    weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1 # macierz o wymiarach: liczba neuronow w warstwie ukrytej X liczba neuronow w warstwie wyjsciowej
    return weights_input_hidden, weights_hidden_output # macierze to losowe liczby z przedziału od -1 do 1

# Podział danych na zbiory treningowe, generalizacyjne i walidacyjne
def split_data(data, train_ratio=0.6, validation_ratio=0.2):
    np.random.shuffle(data) # tasowanie danych
    
    train_size = int(len(data) * train_ratio) 
    validation_size = int(len(data) * validation_ratio)

    train_data = data[:train_size] # wybiera obserwacje do liczby "train_size"
    validation_data = data[train_size:train_size + validation_size] # wybiera obserwacje od "train_size" do sumy "train_size" i "validation_size"
    test_data = data[train_size + validation_size:] # wybiera obserwacje od powyzszej sumy do końca

    return train_data, validation_data, test_data

# # Podziel dane na trzy grupy ze względu na wartość kolumny 'bmi'

# group_1 = data[data['bmi'] <= 25]
# group_2 = data[(data['bmi'] > 25) & (data['bmi'] < 35)]
# group_3 = data[data['bmi'] >= 35]

# # Wybierz odpowiednie kolumny
# selected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

# # Przygotuj dane treningowe i testowe dla każdej grupy

# train_data_group_1 = group_1.sample(frac=0.9, random_state=42)
# test_data_group_1 = group_1.drop(train_data_group_1.index)
# validation_data_group_1 = test_data_group_1.sample(frac=0.2, random_state=42)
# X_train_group_1 = train_data_group_1[selected_columns[:-1]].values
# y_train_group_1 = train_data_group_1[['stroke']].values
# X_test_group_1 = test_data_group_1[selected_columns[:-1]].values
# y_test_group_1 = test_data_group_1[['stroke']].values
# X_validation_group_1 = validation_data_group_1[selected_columns[:-1]].values
# y_validation_group_1 = validation_data_group_1[['stroke']].values

# train_data_group_2 = group_2.sample(frac=0.9, random_state=42)
# test_data_group_2 = group_2.drop(train_data_group_2.index)
# validation_data_group_2 = test_data_group_2.sample(frac=0.2, random_state=42)
# X_train_group_2 = train_data_group_2[selected_columns[:-1]].values
# y_train_group_2 = train_data_group_2[['stroke']].values
# X_test_group_2 = test_data_group_2[selected_columns[:-1]].values
# y_test_group_2 = test_data_group_2[['stroke']].values
# X_validation_group_2 = validation_data_group_2[selected_columns[:-1]].values
# y_validation_group_2 = validation_data_group_2[['stroke']].values

# train_data_group_3 = group_3.sample(frac=0.9, random_state=42)
# test_data_group_3 = group_3.drop(train_data_group_3.index)
# validation_data_group_3 = test_data_group_3.sample(frac=0.2, random_state=42)
# X_train_group_3 = train_data_group_3[selected_columns[:-1]].values
# y_train_group_3 = train_data_group_3[['stroke']].values
# X_test_group_3 = test_data_group_3[selected_columns[:-1]].values
# y_test_group_3 = test_data_group_3[['stroke']].values
# X_validation_group_3 = validation_data_group_3[selected_columns[:-1]].values
# y_validation_group_3 = validation_data_group_3[['stroke']].values

# # Połącz dane zbiorów z poszczególnych grup
# X_test_bmi = np.concatenate([X_test_group_1, X_test_group_2, X_test_group_3])
# y_test_bmi = np.concatenate([y_test_group_1, y_test_group_2, y_test_group_3])

# X_train_bmi = np.concatenate([X_train_group_1, X_train_group_2, X_train_group_3])
# y_train_bmi = np.concatenate([y_train_group_1, y_train_group_2, y_train_group_3])

# X_validation_bmi = np.concatenate([X_validation_group_1, X_validation_group_2, X_validation_group_3])
# y_validation_bmi = np.concatenate([y_validation_group_1, y_validation_group_2, y_validation_group_3])

# Funkcja dostosowująca tempa nauki
def adjust_learning_rate(learning_rate, mse, previous_mse, learning_rate_adjust, threshold=0.001):
    if mse < previous_mse:
        learning_rate *= 1.1  # jeśli błąd maleje, to zwiększa współczynnik uczenia o 10%
    else:
        learning_rate *= 0.5  # jeśli błąd nie maleje, to zmniejszamy współczynnik uczenia o 50%

    if np.abs(mse - previous_mse) < threshold: #jeśli różnica między błędami jest mniejsza niż dany próg to zmniejszamy learning rate, bo zbliżamy się do optymalnej konfiguracji
        learning_rate *= learning_rate_adjust

    return learning_rate

# Trening sieci neuronowej
def train(X, y, learning_rate, learning_rate_adjust, epochs, hidden_size, activation_function):
    input_size = X.shape[1] # liczba neuronów to liczba kolumn 
    output_size = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Przejście w przód
        hidden_layer_input = np.dot(X, weights_input_hidden) # iloczyn macierzy zmiennych X i zainicjalizowanych wag dla neuronów wejściowych do warstwy ukrytej
        hidden_layer_output = activation_function(hidden_layer_input) # zastosowanie funkcji aktywacji 

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) # iloczyn macierzy po zastosowaniu funkcji aktywacji oraz wag dla neuronów wyjściowych z warstwy ukrytej
        predicted_output = activation_function(output_layer_input) # ponowne zastosowanie funkcji aktywacji 

        # Propagacja wsteczna
        output_error = y - predicted_output # błąd predykcji
        output_delta = output_error * activation_function(predicted_output, derivative=True) # delta informuje nas jak na błąd wyjściowy wpływa każda jednostka 
        # pochodna informuje nas o tym, jak szybko zmienia się funkcja aktywacji, co pozwala zrozumieć jak bardzo powinny zmieniać się wagi

        hidden_layer_error = output_delta.dot(weights_hidden_output.T) # iloczyn macierzy błędu delta oraz transponowanej macierzy wag warstwy ukrytej 
        hidden_layer_delta = hidden_layer_error * activation_function(hidden_layer_output, derivative=True) # aktualizacja błędu dla warstwy ukrytej
        # idziemy od warsty wyjściowej, stąd nazwa propagacja wsteczna

        # Aktualizacja wag przez przemnożenie przez macierze otrzymane w propagacji wstecznej oraz learning_rate
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate 
        weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate

        # dostosowanie learning rate na podstawie błędu średniokwadratowego
        learning_rate = adjust_learning_rate(
            learning_rate, 
            np.mean(output_error**2),  #błąd średniokwadratowy
            np.mean((y - predict(X, weights_input_hidden, weights_hidden_output, activation_function))**2), # poprzedni błąd średniokwadratowy
            learning_rate_adjust
        )

    return weights_input_hidden, weights_hidden_output

# Predykcja na podstawie wytrenowanej sieci
def predict(X, weights_input_hidden, weights_hidden_output, activation_function):
    hidden_layer_input = np.dot(X, weights_input_hidden)  # iloczyn zmiennych i wag wejściowych do warstwy ukrytej
    hidden_layer_output = activation_function(hidden_layer_input)  # wartości funkcji aktywacji jako wyjście warstwy ukrytej 

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) #iloczyn wyjścia warstwy ukrytej i wag wyjściowych z warstwy ukrytej
    predicted_output = activation_function(output_layer_input) # ponowne zastosowanie wartości funkcji aktywacji

    return predicted_output

# Funkcja obliczająca błąd predykcji
def calculate_error(predictions, labels):
    return np.mean(np.abs(predictions - labels))

# Funkcja obliczająca procent skuteczności sieci
def correct(y,predict):
    correct_predictions=0
    for i in range(1, len(y)):
        if(abs(y[i]-predict[i])<0.45): # jako poprawne predykcje uznano te, których błąd jest mniejszy niż 0,45
            correct_predictions+=1
    return correct_predictions/len(y) 

# Wczytaj dane z pliku CSV
data = pd.read_csv('healthcare-dataset-stroke-data.csv', delimiter=";")

# Wybierz odpowiednie kolumny
selected_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
data = data[selected_columns]

# Przygotuj dane treningowe
labels = data['stroke'].values.reshape(-1, 1)  # oddzielam zmienną "stroke" i tworze z niej kolumnę
data = data.drop('stroke', axis=1).values # usuwam ze zmienny zmienną "stroke"

# Podziel dane na zbiory
train_data, validation_data, test_data = split_data(np.concatenate((data, labels), axis=1)) # podział danych (połączonych z etykietami)

X_train = train_data[:, :-1] # obcinamy ostatnią kolumnę "stroke"
y_train = train_data[:, -1].reshape(-1, 1) # "stroke" jako kolumna

X_validation = validation_data[:, :-1]
y_validation = validation_data[:, -1].reshape(-1, 1)

X_test = test_data[:, :-1]
y_test = test_data[:, -1].reshape(-1, 1)

# Przykładowe parametry

learning_rates = [0.01]
learning_rate_adjusts = [0.0005]
epochses = [1000]
hidden=[3]
hidden_sizes = [8]
activation_functions = [sigmoid]
repeat = [1]

# # Kod służący do zapisu wyników
# results_df = df = pd.DataFrame({"learning_rate":[1],
#                                 "learning_rate_adjust":[1],
#                                 "epochs": [1],
#                                 "hidden_size": [1],
#                                 "activation_function":["funkcja_aktywacji1"],
#                                 "error_train":[1],
#                                 "error_validation":[1],
#                                 "error_test":[1]})

# # determining the name of the file
# file_name = 'SNN_results.xlsx'

# Testowanie różnych parametrów
for i in repeat:
    for lr in learning_rates:
        for lr_adj in learning_rate_adjusts:
            for epochs in epochses:
                for hidden_size in hidden_sizes:
                    for activation_function in activation_functions:
                        # Trening sieci
                        trained_weights_input_hidden, trained_weights_hidden_output = train(
                            X_train, y_train, lr, lr_adj, epochs, hidden_size, activation_function
                        )

                        # Predykcja dla danych treningowych
                        predictions_train = predict(X_train, trained_weights_input_hidden, trained_weights_hidden_output, activation_function)
                        
                        # Predykcja dla danych walidacyjnych
                        predictions_validation = predict(X_validation, trained_weights_input_hidden, trained_weights_hidden_output, activation_function)
                        
                        # Predykcja dla danych testowych
                        predictions_test = predict(X_test, trained_weights_input_hidden, trained_weights_hidden_output, activation_function)
                        
                        # # Oblicz błędy predykcji
                        # error_train = calculate_error(predictions_train, y_train)
                        # error_validation = calculate_error(predictions_validation, y_validation)
                        # error_test = calculate_error(predictions_test, y_test)

                        # results_df = results_df._append(({"learning_rate":lr,
                        #                 "learning_rate_adjust":lr_adj,
                        #                 "epochs": epochs,
                        #                 "hidden_size": hidden_size,
                        #                 "activation_function": activation_function.__name__,
                        #                 "error_train": error_train,
                        #                 "error_validation": error_validation,
                        #                 "error_test": error_test}), ignore_index=True)

                        #Sprawdzanie skuteczności sieci

                        correct_pred=correct(y_test,predictions_test)

                        print("Liczba warstw ukrytych:", hidden)
                        print("Liczba neuronów w pojedynczej warstwie ukrytej:", hidden_size)
                        print("Learning rate:", lr)
                        print("Learning rate adjust:", lr_adj)
                        print("Correct predictions:", correct_pred)
                        # print("Error (Train):", error_train)
                        # print("Error (Validation):", error_validation)
                        # print("Error (Test):", error_test)
                        print("-----")

                        # # creating an ExcelWriter object
                        # with pd.ExcelWriter(file_name) as writer:
                        #     # writing to the 'Result_1' sheet
                        #     results_df.to_excel(writer, sheet_name='Result_1', index=False)
                        # print('DataFrames are written to Excel File successfully.')
