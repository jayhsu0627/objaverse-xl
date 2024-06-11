import csv
import os

def check_repeated_lines(filename):
    """
    Check for repeated lines in a text file, ignoring the line numbers.

    Parameters:
    filename (str): The name of the text file to check.

    Returns:
    None
    """
    # Dictionary to store the count of each line's content (excluding line numbers)
    line_counts = {}

    # Read the file and count occurrences of each line's content
    with open(filename, 'r') as file:
        for line in file:
            line_content = ' '.join(line.strip().split(' ')[1:])  # Ignore the line number
            if line_content in line_counts:
                line_counts[line_content] += 1
            else:
                line_counts[line_content] = 1

    # Find and print repeated lines
    repeated_lines = {line: count for line, count in line_counts.items() if count > 1}

    if repeated_lines:
        print("Repeated lines:")
        for line, count in repeated_lines.items():
            print(f'"{line}" is repeated {count} times')
    else:
        print("No repeated lines found")


def remove_repeated_lines(input_filename, output_filename):
    """
    Remove repeated lines from a text file and save the result to a new file.

    Parameters:
    input_filename (str): The name of the input text file.
    output_filename (str): The name of the output text file.

    Returns:
    None
    """
    # Dictionary to store the count of each line (excluding the first column)
    line_counts = {}

    # Read the input file and count occurrences of each line
    with open(input_filename, 'r') as input_file:
        for line in input_file:
            line = line.strip()
            line_key = line.split(maxsplit=1)[1]  # Exclude the first column
            if line_key in line_counts:
                line_counts[line_key] += 1
            else:
                line_counts[line_key] = 1

    # Write unique lines to the output file
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            line_key = line.split(maxsplit=1)[1]  # Exclude the first column
            if line_counts[line_key] == 1:
                output_file.write(f"{line}\n")

def rename_line(input_filename, output_filename):
    # Write unique lines to the output file
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        print('here')
        print(input_filename, output_filename)

        writer = csv.writer(output_file, delimiter='\t')

        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            # print(line_number)
            line_key = line.split(maxsplit=1)[1]  # Exclude the first column
            if line_number>1:
                output_file.write(f"{line_number-1}\t{line_key}\n")
                # writer.writerow([str(line_number), line_key])

            else:
                output_file.write(f"ID\t{line_key}\n")
    
                # Write the header
                # writer.writerow(["ID", line_key])

# Example usage
input_filename = '/fs/nexus-scratch/sjxu/objaverse-xl/scripts/rendering/meta.txt'  # Replace with your text file name
temp_filename = './temp.txt'
output_filename = './example_unique.txt'  # Replace with the desired output file name
remove_repeated_lines(input_filename, temp_filename)
rename_line(temp_filename, output_filename)
os.remove(temp_filename)

print(f"Unique lines saved to {output_filename}")

# Example usage
filename = output_filename
check_repeated_lines(filename)
