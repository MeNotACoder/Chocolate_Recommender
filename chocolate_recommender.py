import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

# Load the dataset
chocolate_data = pd.read_csv("chocolate.csv")

# Split the dataset into features and labels
X = chocolate_data.iloc[:, :-1]
y = chocolate_data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network with 2 hidden layers
clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

# Train the model on the training data
clf.fit(X_train, y_train)

# Save the trained model
filename = "chocolate_model.sav"
pickle.dump(clf, open(filename, "wb"))

# Create the GUI
root = tk.Tk()
root.title("Chocolate Decider")

# Function to predict if the chocolate is good or bad
def predict():
    # Get the user inputs
    is_dark = int(dark_var.get())
    is_cheap = int(cheap_var.get())
    is_branded = int(branded_var.get())
    user_data = [is_dark, is_cheap, is_branded]

    # Load the trained model
    loaded_model = pickle.load(open(filename, "rb"))

    # Make a prediction
    prediction = loaded_model.predict([user_data])[0]

    # Show the prediction in a messagebox
    if prediction == 1:
        messagebox.showinfo("Prediction", "This chocolate is GOOD!")
    else:
        messagebox.showinfo("Prediction", "This chocolate is BAD :(")

# Function to add user data to the dataset
# Function to add user data to the dataset
def add_data():
    # Get the user inputs
    global chocolate_data
    is_dark = int(dark_var.get())
    is_cheap = int(cheap_var.get())
    is_branded = int(branded_var.get())
    is_good = int(good_var.get())
    user_data = [is_dark, is_cheap, is_branded, is_good]

    # Add the user data to the dataset
    new_data = pd.DataFrame([user_data], columns=["dark", "cheap", "branded", "good"])
    chocolate_data = pd.concat([chocolate_data, new_data], ignore_index=True)

    # Save the updated dataset
    chocolate_data.to_csv("chocolate.csv", index=False)

    # Show a message that the data was added
    messagebox.showinfo("Data Added", "Your data was successfully added to the dataset!")

# Create the GUI widgets
dark_var = tk.StringVar(value="0")
cheap_var = tk.StringVar(value="0")
branded_var = tk.StringVar(value="0")
good_var = tk.StringVar(value="0")

dark_label = tk.Label(root, text="Is the chocolate dark? (0/1)")
dark_entry = tk.Entry(root, textvariable=dark_var)

cheap_label = tk.Label(root, text="Is the chocolate cheap? (0/1)")
cheap_entry = tk.Entry(root, textvariable=cheap_var)

branded_label = tk.Label(root, text="Is the chocolate branded? (0/1)")
branded_entry = tk.Entry(root, textvariable=branded_var)

predict_button = tk.Button(root, text="Predict", command=predict)

good_label = tk.Label(root, text="Is the chocolate good? (0/1)")
good_entry = tk.Entry(root, textvariable=good_var)

add_button = tk.Button(root, text="Add Data", command=add_data)

# Place the GUI widgets on the
root.geometry("400x200")

dark_label.grid(row=0, column=0)
dark_entry.grid(row=0, column=1)

cheap_label.grid(row=1, column=0)
cheap_entry.grid(row=1, column=1)

branded_label.grid(row=2, column=0)
branded_entry.grid(row=2, column=1)

predict_button.grid(row=3, column=0, columnspan=2)

good_label.grid(row=4, column=0)
good_entry.grid(row=4, column=1)

add_button.grid(row=5, column=0, columnspan=2)


root.mainloop()
