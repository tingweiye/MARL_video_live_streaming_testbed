import os
print(os.getcwd())
path = "../data"

for idx in range(600):
    for rate in [0.3, 0.5, 1.0, 2.0, 3.0, 6.0]:
        numBytes = rate * 1e6 / 8
        file_name = str(idx) + '_' + str(rate) + '.mp4'
        with open(os.path.join(path, file_name), 'w') as f:
            f.write('a' * int(numBytes))
