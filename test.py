import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from PIL import ImageTk, Image
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def imagecall(k):
    img = Image.open(k)
    img = img.resize((280, 280), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(image_frame, image=img)
    panel.image = img  # keep a reference to the PhotoImage object to prevent it from being garbage collected
    panel.pack(side='top', pady=10)

# Define a function to handle the "Browse" button click event
def browse_file():
    if file_path := filedialog.askopenfilename():
        # Display the selected image in the GUI
        imagecall(file_path)
        # Load the image and preprocess it
        img = load_img(file_path, target_size=(32, 32))
        img_array = img_to_array(img)
        img_array /= 255.
        img_array = np.expand_dims(img_array, axis=0)
        # Make a prediction using the model
        prediction = model.predict(img_array)
        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        # Map the predicted class index to the class name
        predicted_class_name = [k for k, v in class_map.items() if v == predicted_class_index]
        if predicted_class_name:
            predicted_class_name = predicted_class_name[0]
        else:
            predicted_class_name = "Unknown"
        # Update the label with the predicted class name
        predicted_label.config(text=f"Predicted class: {predicted_class_name}")
        # Display a histogram of the predicted class probabilities
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.bar(range(len(prediction[0])), prediction[0])
        ax.set_xticks(range(len(prediction[0])))
        ax.set_xticklabels(class_map.keys(), rotation=90)
        ax.set_title('Predicted class probabilities')
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

# Define a function to handle the "Clear" button click event
def clear_image():
    # Remove the selected image from the GUI
    for widget in image_frame.winfo_children():
        widget.destroy()
    # Remove the graph from the GUI
    for widget in graph_frame.winfo_children():
        widget.destroy()
    # Reset the predicted class label
    predicted_label.config(text="Predicted class:")

# Define the class map
class_map = {
     'character_1_ka': 0,
    'character_2_kha': 1,
    'character_3_ga': 2,
    'character_4_gha': 3,
    'character_5_kna': 4,
    'character_6_cha': 5,
    'character_7_chha': 6,
    'character_8_ja': 7,
    'character_9_jha': 8,
    'character_10_yna': 9,
    'character_11_taamatar': 10,
    'character_12_thaa': 11,
    'character_13_daa': 12,
    'character_14_dhaa': 13,
    'character_15_adna': 14,
    'character_16_tabala': 15,
    'character_17_tha': 16,
    'character_18_da': 17,
    'character_19_dha': 18,
    'character_20_na': 19,
    'character_21_pa': 20,
    'character_22_pha': 21,
    'character_23_ba': 22,
    'character_24_bha': 23,
    'character_25_ma': 24,
    'character_26_yaw': 25,
    'character_27_ra': 26,
    'character_28_la': 27,
    'character_29_waw': 28,
    'character_30_motosaw': 29,
    'character_31_petchiryakha': 30,
    'character_32_patalosaw': 31,
    'character_33_ha': 32,
    'character_34_chhya': 33,
    'character_35_tra': 34,
    'character_36_gya': 35,
    'digit_0': 36,
    'digit_1': 37,
    'digit_2': 38,
    'digit_3': 39,
    'digit_4': 40,
    'digit_5': 41,
    'digit_6': 42,
    'digit_7': 43,
    'digit_8': 44,
    'digit_9': 45,
}

# Load the trained model
model = tf.keras.models.load_model('new.h5')

# Create the main window of the GUI
window = tk.Tk()
window.title("Devanagari Character Recognition")
window.geometry('800x500')

# Create the "Browse" button
browse_button = tk.Button(window, text="Browse", command=browse_file)
browse_button.grid(row=0, column=0, pady=10)

# Create the "Clear" button
clear_button = tk.Button(window, text="Clear", command=clear_image)
clear_button.grid(row=0, column=1, pady=10)

# Create a frame to hold the selected image
image_frame = tk.Frame(window)
image_frame.grid(row=1, column=0, padx=10, pady=10)

# Create a frame to hold the graph
graph_frame = tk.Frame(window)
graph_frame.grid(row=1, column=1, padx=10, pady=10)

# Create a label to show the predicted class
predicted_label = tk.Label(window, text="Predicted class:")
predicted_label.grid(row=2, column=0, columnspan=2, pady=10)

# Start the GUI event loop
window.mainloop()