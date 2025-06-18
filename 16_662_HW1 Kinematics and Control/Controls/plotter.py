import csv
import matplotlib.pyplot as plt

# change csv files, force in 1 and impedance in 2

# Read data from the first CSV file
with open('force_vs_time_32.csv', 'r') as file:
    reader = csv.reader(file)
    data1 = []
    for line in reader:
            data1.append((float(line[0]), float(line[1])))



# Read data from the second CSV file
with open('impedance_vs_time_32.csv', 'r') as file:
    reader = csv.reader(file)
    data2 = []
    for line in reader:
            data2.append((float(line[0]), float(line[1])))

# Extract x and y values from the data
x1, y1 = zip(*data1)
x2, y2 = zip(*data2)

# Plot the data and add labels, title, and legend
plt.plot(x1, y1, label='Force Control')
plt.plot(x2, y2, label='Impedance Control')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Plot of Force and Impedance Control for Oscillating Board')
plt.legend()
plt.show()