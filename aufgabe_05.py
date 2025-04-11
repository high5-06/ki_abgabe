import tensorflow as tf        # Für das Laden und Verwenden des KI-Modells
import numpy as np             # Für Datenverarbeitung und Arrays
import string                  # Für Buchstaben A-Z
import cv2                     # Für das Laden und Bearbeiten von Bildern
import os                      # Für Datei- und Ordnernavigation

# Funktion zum Laden der Bilddaten aus Ordnerstruktur A-Z
def load_data(input_folder, size=(28, 28)):
    images = []  # Hier werden alle Bilder gespeichert
    labels = []  # Passende Labels (0=A, 1=B, ...) zu den Bildern

    # Jeden Buchstaben-Ordner (A-Z) im Dataset durchgehen
    for label, letter in enumerate(sorted(os.listdir(input_folder))):
        letter_path = os.path.join(input_folder, letter)

        if os.path.isdir(letter_path):  # Nur Ordner, keine Dateien
            for filename in os.listdir(letter_path):
                if filename.lower().endswith(".png"):  # Nur .png-Bilder laden
                    img_path = os.path.join(letter_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Bild in Graustufen laden
                    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # Auf 28x28 skalieren
                    img_array = 1 - img_resized / 255.0  # Normalisieren und Farben invertieren (weiß auf schwarz)
                    images.append(img_array)
                    labels.append(label)

    # In NumPy-Arrays umwandeln und für das CNN in Form bringen
    x_data = np.array(images).reshape(-1, 28, 28, 1).astype('float32')
    y_data = np.array(labels).astype('int')
    return x_data, y_data

# Wandelt Vorhersage (Index) zurück in Buchstabe (z. B. 0 -> A)
def index_to_letter(index):
    return string.ascii_uppercase[index]

# Modellvorhersagen berechnen und in Buchstaben umwandeln
def predict_letters(model, x_test):
    predictions = model.predict(x_test)                      # Wahrscheinlichkeiten für jede Klasse
    predicted_indices = np.argmax(predictions, axis=1)       # Index der höchsten Wahrscheinlichkeit
    predicted_letters = [index_to_letter(idx) for idx in predicted_indices]  # In Buchstaben umwandeln
    return predicted_letters

# Hauptprogramm
if __name__ == "__main__":
    # Vortrainiertes Modell laden
    model = tf.keras.models.load_model("verbessertes_cnn_model.keras")

    # Bilddaten aus A–Z-Ordnerstruktur laden
    x, y = load_data("C:/Users/User/OneDrive - HTL Anichstrasse/Desktop/Schule/Schuljahr24_25/KI_Korber/Perceptron/Level 1/.venv/BigDataSet")
    
    # 10% der Daten als Testdaten nehmen
    from sklearn.model_selection import train_test_split
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Modellvorhersagen auf Testdaten berechnen
    predicted_letters = predict_letters(model, x_test)

    # Ausgabe der ersten 10 Vorhersagen im Vergleich zur Wahrheit
    print("\nManuelle Überprüfung der Vorhersagen:")
    for i in range(10):
        print(f"Bild {i+1}: Richtig = {index_to_letter(y_test[i])}, Vorhersage = {predicted_letters[i]}")
