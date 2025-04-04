import csv
import numpy as np

import webcolors

def hex_to_color_name(hex_value):
    """
    Convert a hex color string to its closest CSS3 color name.

    Parameters:
    hex_value (str): A hex color string in the format '#RRGGBB' or '#RRGGBBAA'.

    Returns:
    str: The closest CSS3 color name.
    """
    try:
        # Get the closest CSS3 color name
        color_name = webcolors.hex_to_name(hex_value, spec='css3')
    except ValueError:
        # If the color name is not found, find the closest color
        rgb_value = webcolors.hex_to_rgb(hex_value)
        closest_name = None
        min_distance = float('inf')
        
        for name, rgb in webcolors._definitions._CSS3_NAMES_TO_HEX.items():
        # for name, rgb in webcolors._defin._CSS3_NAMES_TO_HEX.items():
            candidate_rgb = webcolors.hex_to_rgb(rgb)
            distance = sum((component1 - component2) ** 2 for component1, component2 in zip(rgb_value, candidate_rgb))
            if distance < min_distance:
                closest_name = name
                min_distance = distance
        
        color_name = closest_name
    
    return color_name

def rgba_to_hex(rgba):
    """
    Convert an RGBA color tuple to a hexadecimal string.

    Parameters:
    rgba (tuple): A tuple containing four floats or integers (R, G, B, A) each in the range 0-255.

    Returns:
    str: A hexadecimal color string in the format '#RRGGBBAA'.
    """
    # Ensure all components are integers in the range 0-255
    r, g, b, _ = (int(c*255) for c in rgba)

    # Format as hexadecimal
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_rgb_color(mean=1.0, std_dev=1/3):
    """
    Generate an RGB color that follows a normal distribution
    with the mean color close to white (1, 1, 1).

    Parameters:
    mean (float): The mean of the normal distribution. Default is 1.0.
    std_dev (float): The standard deviation of the normal distribution. Default is 1/3.

    Returns:
    tuple: A tuple representing the RGB color.
    """
    # Generate RGB values from a normal distribution
    r, g, b = np.random.normal(loc=mean, scale=std_dev, size=3)

    # Clamp the values between 0 and 1
    r = round(np.clip(r, 0, 1),2)
    g = round(np.clip(g, 0, 1),2)
    b = round(np.clip(b, 0, 1),2)
    a = round(1.0,2)

    return (r, g, b, a)

# Sample data
data = [
    {"ID": 1, "words": "example", "gloss": "a representative form or pattern", "num_train_images": 5},
    {"ID": 2, "words": "test", "gloss": "a procedure for critical evaluation", "num_train_images": 3},
    {"ID": 3, "words": "script", "gloss": "a written version of a play or other dramatic composition", "num_train_images": 8},
]


data = []
light_type = ["POINT", "SUN", "SPOT", "AREA"]

rotation_n = 2
color_n = 2
num = 0
for i in range(rotation_n):
    rotation_array = (
        round(np.radians(np.random.uniform(-360, 360)), 4),
        round(np.radians(np.random.uniform(-360, 360)), 4),
        round(np.radians(np.random.uniform(-360, 360)), 4)
    )
    for j in range(color_n):
        # color_array = (
        #     round(np.random.random(),2),
        #     round(np.random.random(),2),
        #     round(np.random.random(),2),
        #     1,
        # )
        color_array = generate_rgb_color()
        color_string = hex_to_color_name(rgba_to_hex(color_array))
        # print(color_string)
        for k in range(len(light_type)):
            temp_key = {}
            temp_key["ID"] = num
            temp_key["words"] = str(light_type[k]).lower() +", rotation at " + str(rotation_array) + ", color is " +str(color_array)
            temp_key["gloss"] = "A " + color_string + " color "+ str(light_type[k]).lower() +" light"
            temp_key["light_type"] = str(light_type[k])
            temp_key["rotation"] = rotation_array
            temp_key["color"] = color_array
            temp_key["num_train_images"] = 1
            data.append(temp_key)
            num +=1

# Specify the filename
filename = 'meta.txt'

# Create the text table
with open(filename, 'w', newline='') as txtfile:
    writer = csv.writer(txtfile, delimiter='\t')
    
    # Write the header
    writer.writerow(["ID", "words", "gloss", "light_type", "rotation", "color", "num_train_images"])
    
    # Write the data
    for entry in data:
        writer.writerow([entry["ID"], entry["words"], entry["gloss"], entry["light_type"], entry["rotation"], entry["color"], entry["num_train_images"]])

print(f"Table written to {filename}")
