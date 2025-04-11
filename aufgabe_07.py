import tkinter as tk  # GUI-Toolkit für Fenster & Buttons
from PIL import Image, ImageDraw, ImageOps  # Für Bilderstellung & -bearbeitung
import numpy as np  # Für Datenumwandlung und Mathe
from tensorflow.keras.models import load_model  # Modell laden

# Vortrainiertes Modell laden
model = load_model("verbessertes_cnn_model.keras")  

# Funktion: Gibt Buchstabe mit höchster Wahrscheinlichkeit zurück
def predict_letter(probabilities):
    index = np.argmax(probabilities)  # Index des höchsten Werts
    return chr(index + ord('A'))  # 0 -> A, 1 -> B, ...

# GUI-Klasse
class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Buchstabenerkennung")  # Fenster-Titel

        # Größe des Zeichenbereichs
        self.canvas_width = 200
        self.canvas_height = 200

        # Zeichenfläche (weiß)
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Button: Erkennen
        self.button_predict = tk.Button(master, text="Erkennen", command=self.predict_drawing)
        self.button_predict.pack()

        # Button: Löschen
        self.button_clear = tk.Button(master, text="Löschen", command=self.clear_canvas)
        self.button_clear.pack()

        # Textfeld für das Ergebnis
        self.label_result = tk.Label(master, text="Erkannt: ")
        self.label_result.pack()

        # Graustufenbild zum parallelen Mitzeichnen
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Beim Zeichnen mit gedrückter Maustaste wird paint() aufgerufen
        self.canvas.bind("<B1-Motion>", self.paint)

    # Beim Zeichnen mit Maus wird schwarzer Punkt auf Canvas + Bild gezeichnet
    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Radius des Pinsels
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")  # Zeichne auf Canvas
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)  # Zeichne auf Bild (für Modell)

    # Vorhersage durchführen, wenn Button "Erkennen" gedrückt wird
    def predict_drawing(self):
        # Bild auf 28x28 skalieren (wie fürs Modell nötig)
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # In NumPy-Array umwandeln, normalisieren und invertieren (0=weiß → 1=schwarz)
        image_array = 1 - np.array(resized_image).astype(np.float32) / 255.0
        image_array = image_array.reshape(1, 28, 28)  # Für das CNN vorbereiten (Batch-Größe 1)

        # Modellvorhersage
        prediction = model.predict(image_array)
        letter = predict_letter(prediction[0])  # Buchstabe aus Vorhersage ableiten
        prob = model.

        # Ergebnis im Fenster anzeigen
        self.label_result.config(text=f"Erkannt: {letter}")

    # Canvas und Bild löschen
    def clear_canvas(self):
        self.canvas.delete("all")  # GUI leeren
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)  # Weißes Bild
        self.draw = ImageDraw.Draw(self.image)

# Startet das Programm
root = tk.Tk()           # Fenster erstellen
app = DrawingApp(root)   # App starten
root.mainloop()          # Endlosschleife der GUI
