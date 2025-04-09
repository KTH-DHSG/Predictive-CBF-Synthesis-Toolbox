"""

    This module provides a function to print warnings in a true-color orange format.
    It uses ANSI escape codes to achieve this effect.

    The function `colorWarning` takes a message as input and prints it in orange.

    (c) Adrian Wiltz, 2025

"""


import warnings

# ANSI escape code for true-color orange
ORANGE_COLOR = "\033[38;2;255;165;0m"
DARK_ORANGE = "\033[38;2;255;140;0m"
RESET = "\033[0m"

# Function to print a true-color orange warning
def colorWarning(message):
    print(f"{DARK_ORANGE}WARNING: {message}{RESET}")