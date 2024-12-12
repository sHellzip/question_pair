import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

# Function to process and analyze texts
def analyze_texts():
    text1 = text1_entry.get("1.0", tk.END).strip()
    text2 = text2_entry.get("1.0", tk.END).strip()

    # Apply options
    if remove_numbers_var.get():
        text1 = re.sub(r'\d+', '', text1)
        text2 = re.sub(r'\d+', '', text2)
    if remove_punctuation_var.get():
        text1 = re.sub(r'[\W_]+', ' ', text1)
        text2 = re.sub(r'[\W_]+', ' ', text2)
    if convert_lowercase_var.get():
        text1 = text1.lower()
        text2 = text2.lower()

    # Calculate statistics
    words1 = text1.split()
    words2 = text2.split()

    word_count1 = len(words1)
    word_count2 = len(words2)

    unique_words1 = set(words1)
    unique_words2 = set(words2)

    common_words = unique_words1 & unique_words2
    similarity_score = len(common_words) / max(len(unique_words1 | unique_words2), 1)

    # Update results area
    result_text.set(f"Similarity Score: {similarity_score:.4f}\n\n" +
                    f"Text 1 Statistics:\n- Word count: {word_count1}\n- Unique words: {len(unique_words1)}\n\n" +
                    f"Text 2 Statistics:\n- Word count: {word_count2}\n- Unique words: {len(unique_words2)}\n\n" +
                    f"Common unique words: {len(common_words)}")

    # Update charts
    update_charts(similarity_score, word_count1, word_count2, len(common_words), Counter(words1 + words2).most_common(5))

def update_charts(similarity, text1_words, text2_words, common_words, top_words):
    # Clear previous plots
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # Create new charts
    fig, axes = plt.subplots(3, 1, figsize=(5, 8))

    # Similarity Score
    axes[0].bar(["Similarity"], [similarity], color="skyblue")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Similarity Score")

    # Word Count Comparison
    axes[1].bar(["Text 1 Words", "Text 2 Words", "Common Words"], [text1_words, text2_words, common_words], color=["blue", "green", "red"])
    axes[1].set_title("Word Count Comparison")

    # Top 5 Common Words
    words, frequencies = zip(*top_words) if top_words else ([], [])
    axes[2].barh(words, frequencies, color="coral")
    axes[2].set_title("Top 5 Common Words")

    # Render charts in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()
    canvas.draw()

# Main GUI window
root = tk.Tk()
root.title("Enhanced Text Similarity Analysis")

# Text input frames
input_frame = ttk.Frame(root)
input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

chart_frame = ttk.Frame(root)
chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Input widgets
ttk.Label(input_frame, text="Text Input", font=("Arial", 16)).pack(anchor=tk.W)

text1_label = ttk.Label(input_frame, text="Text 1:")
text1_label.pack(anchor=tk.W)

text1_entry = tk.Text(input_frame, height=5, width=40)
text1_entry.pack()

text2_label = ttk.Label(input_frame, text="Text 2:")
text2_label.pack(anchor=tk.W)

text2_entry = tk.Text(input_frame, height=5, width=40)
text2_entry.pack()

# Analysis options
options_frame = ttk.LabelFrame(input_frame, text="Analysis Options")
options_frame.pack(fill=tk.X, pady=10)

remove_numbers_var = tk.BooleanVar(value=True)
remove_punctuation_var = tk.BooleanVar(value=True)
convert_lowercase_var = tk.BooleanVar(value=True)

remove_numbers_check = ttk.Checkbutton(options_frame, text="Remove Numbers", variable=remove_numbers_var)
remove_numbers_check.pack(anchor=tk.W)

remove_punctuation_check = ttk.Checkbutton(options_frame, text="Remove Punctuation", variable=remove_punctuation_var)
remove_punctuation_check.pack(anchor=tk.W)

convert_lowercase_check = ttk.Checkbutton(options_frame, text="Convert to Lowercase", variable=convert_lowercase_var)
convert_lowercase_check.pack(anchor=tk.W)

# Analyze button
analyze_button = ttk.Button(input_frame, text="Analyze Texts", command=analyze_texts)
analyze_button.pack(pady=10)

# Results area
result_text = tk.StringVar()
result_label = ttk.Label(input_frame, textvariable=result_text, justify=tk.LEFT, background="white", anchor=tk.N)
result_label.pack(fill=tk.BOTH, expand=True, pady=10)
result_label.config(relief="sunken", padding=5)

# Run the application
root.mainloop()
