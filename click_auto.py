import pyautogui
import time
import threading
from pynput import mouse, keyboard

# Initialize the coordinates to None
x1, y1, x2, y2 = None, None, None, None

# Define a function to handle the mouse events
def on_click(x, y, button, pressed):
    global x1, y1, x2, y2
    if pressed and button == mouse.Button.left:
        if x1 is None:
            x1, y1 = x, y
            print("First location registered at ({}, {})".format(x1, y1))
        elif x2 is None:
            x2, y2 = x, y
            print("Second location registered at ({}, {})".format(x2, y2))
            return False  # Stop listener after registering the second coordinate

# Define a function to handle the keyboard events
def on_press(key):
    global stop_flag
    if key == keyboard.Key.space:  # Check if the space key is pressed
        print("Terminating the program...")
        stop_flag = True  # Set stop flag to True if the space key is pressed
        return False  # Stop listener if the space key is pressed

# Start the mouse listener in a separate thread
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

# Start the keyboard listener in a separate thread
keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

# Wait for the user to register the coordinates
while x2 is None:
    time.sleep(1)

# Click in the registered coordinates every second
stop_flag = False
while not stop_flag:
    pyautogui.click(x=x1, y=y1)
    pyautogui.click(x=x2, y=y2)
    time.sleep(1)

# Stop the listeners
mouse_listener.stop()
keyboard_listener.stop()
 