import numpy as np

numbers = np.arange(1,101)

sum = np.sum(numbers)
average = np.sum(numbers)/len(numbers)

squareddiff = np.sum((numbers-average)**2) 
std = np.sqrt(squareddiff/(len(numbers)))

print("Standard deviation:")
print(std)

oddnumbers = numbers[::2]
evennumbers = numbers[1::2]

oddaverage = np.sum(oddnumbers)/len(oddnumbers)

oddsquareddiff = np.sum((oddnumbers-oddaverage)**2) 
oddstd = np.sqrt(oddsquareddiff/(len(oddnumbers)))

evenaverage = np.sum(evennumbers)/len(evennumbers)

evensquareddiff = np.sum((evennumbers-evenaverage)**2) 
evenstd = np.sqrt(evensquareddiff/(len(evennumbers)))

print("Standard deviation of odd numbers is:")
print(oddstd)

print("Standard deviation of even numbers is:")
print(evenstd)
