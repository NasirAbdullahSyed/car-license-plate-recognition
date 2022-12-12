import string
import random


def num_generator(size=7, chars=string.ascii_uppercase+string.digits+" "+"-"+"."):
    return ''.join(random.choice(chars) for _ in range(size))


dictionary = []

for i in range(500):
    plate = num_generator(3)
    dictionary.append(plate)

for i in range(500):
    plate = num_generator(4)
    dictionary.append(plate)

for i in range(500):
    plate = num_generator(5)
    dictionary.append(plate)

for i in range(500):
    plate = num_generator(6)
    dictionary.append(plate)

for i in range(1000):
    plate = num_generator(7)
    dictionary.append(plate)

for i in range(1000):
    plate = num_generator(8)
    dictionary.append(plate)

for i in range(500):
    plate = num_generator(9)
    dictionary.append(plate)

for i in range(500):
    plate = num_generator(10)
    dictionary.append(plate)

with open("dictionary1.txt", "w") as outfile:
    outfile.write("\n".join(dictionary))
