import tensorflow
import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist  # Import des MNIST-Datensatzes

# Laden des trainierten CNN-Modells
model = load_model('mnist_model.h5')

# Funktion zum Zeichnen der Zahl
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill='white')

# Funktion zum Suchen der gezeichneten Zahl
def search_number():
    # Screenshot des Canvas erstellen
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab(bbox=(x, y, x1, y1)).resize((28, 28))

    # Vorverarbeitung des Bildes
    gray = np.array(image.convert('L'))
    normalized = gray.astype('float32') / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))

    # Vorhersage der Zahl
    prediction = model.predict(reshaped)
    predicted_number = np.argmax(prediction)

    # Anzeigen der Vorhersage
    result_label.config(text="Erkannte Zahl: {}".format(predicted_number))

# Funktion zum Löschen des gezeichneten Bildes
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Erkannte Zahl: -")

# Funktion zum Berichtigen der erkannten Zahl
def correct_number(is_correct):
    global correction_popup, second_correction_popup
    if is_correct:
        # Pop-up anzeigen
        correction_popup = Toplevel(root)
        correction_popup.title("Berichtigung")
        correction_popup.geometry("200x100")
        correction_popup.configure(bg="white")

        correction_label = Label(correction_popup, text="Vielen Dank für dein Feedback!", font=("Arial", 12), bg="white")
        correction_label.pack(pady=10)

        close_button = Button(correction_popup, text="Schließen", command=lambda: close_popup(correction_popup), font=("Arial", 12), bg="#2196F3", fg="white", relief="groove")
        close_button.pack()

    else:
        # Pop-up für Falsch erkannt anzeigen
        correction_popup = Toplevel(root)
        correction_popup.title("Berichtigung")
        correction_popup.geometry("250x150")
        correction_popup.configure(bg="white")

        correction_label = Label(correction_popup, text="Bitte richtige Zahl eingeben:", font=("Arial", 12), bg="white")
        correction_label.pack(pady=10)

        correction_entry = Entry(correction_popup, font=("Arial", 16), justify="center")
        correction_entry.pack(pady=10)

        def continue_correction():
            global corrected_number
            corrected_number = int(correction_entry.get())

            # Pop-up für zweite Berichtigung
            second_correction_popup = Toplevel(root)
            second_correction_popup.title("Berichtigung")
            second_correction_popup.geometry("300x150")
            second_correction_popup.configure(bg="white")

            correction_label = Label(second_correction_popup, text="Dein Ergebnis wurde erfolgreich zu {} berichtigt.".format(corrected_number), font=("Arial", 12), bg="white")
            correction_label.pack(pady=10)

            close_button = Button(second_correction_popup, text="Schließen", command=lambda: close_popups([correction_popup, second_correction_popup]), font=("Arial", 12), bg="#2196F3", fg="white", relief="groove")
            close_button.pack()

            # Trainiere das Modell mit den berichtigten Zahlen
            global x_train, y_train
            corrected_image = cv2.resize(np.array(canvas_image.convert('L')), (28, 28))
            x_train = np.append(x_train, [corrected_image.reshape(28, 28, 1)], axis=0)
            y_train = np.append(y_train, [corrected_number], axis=0)
            y_train = to_categorical(y_train, num_classes=10)
            model.fit(x_train, y_train, epochs=1, batch_size=32)

        continue_button = Button(correction_popup, text="Fortfahren", command=continue_correction, font=("Arial", 12), bg="#2196F3", fg="white", relief="groove")
        continue_button.pack()

# Funktion zum Schließen des Pop-ups
def close_popup(popup):
    popup.destroy()

# Funktion zum Schließen der Pop-ups
def close_popups(popups):
    for popup in popups:
        popup.destroy()

# Funktion zum Schließen des Fensters
def close_window():
    root.destroy()

# Erstellen des Hauptfensters
root = Tk()
root.title("Zahlenerkennung")
root.geometry("400x600")
root.configure(bg="white")

# Erstellen des Zeichenbereichs
canvas = Canvas(root, width=280, height=280, bg='white', bd=5, relief="solid")
canvas.place(x=60, y=60)
canvas.bind("<B1-Motion>", draw)

# Erstellen des Suchen-Buttons
search_button = Button(root, text="Suchen", command=search_number, font=("Arial", 14), bg="#4CAF50", fg="white", relief="groove")
search_button.place(x=150, y=10)

# Erstellen des Labels für die Vorhersage
result_label = Label(root, text="Erkannte Zahl: -", font=("Arial", 16), bg="white")
result_label.place(x=150, y=440)

# Erstellen des Bereinigen-Buttons
clear_button = Button(root, text="Ergebnis bereinigen", command=clear_canvas, font=("Arial", 14), bg="#FF5722", fg="white", relief="groove")
clear_button.place(x=120, y=380)

# Erstellen des Buttons für "Richtig erkannt"
correct_button = Button(root, text="Richtig erkannt", command=lambda: correct_number(True), font=("Arial", 14), bg="#2196F3", fg="white", relief="groove")
correct_button.place(x=60, y=500)

# Erstellen des Buttons für "Falsch erkannt"
wrong_button = Button(root, text="Falsch erkannt", command=lambda: correct_number(False), font=("Arial", 14), bg="#2196F3", fg="white", relief="groove")
wrong_button.place(x=220, y=500)

# Erstellen des Schließen-Buttons
close_button = Button(root, text="X", command=close_window, font=("Arial", 14), bg="red", fg="white", relief="groove", activeforeground="white", activebackground="red")
close_button.place(x=280, y=380)

# Funktion zum Ändern des Schließen-Button-Textes beim Überfahren mit der Maus
def change_close_button_text(event):
    close_button.config(text="Schließen")

# Funktion zum Zurücksetzen des Schließen-Button-Textes beim Verlassen mit der Maus
def reset_close_button_text(event):
    close_button.config(text="X")

# Binden der Funktionen an die Mausereignisse
close_button.bind("<Enter>", change_close_button_text)
close_button.bind("<Leave>", reset_close_button_text)

# Hauptfenster anzeigen
root.mainloop()

