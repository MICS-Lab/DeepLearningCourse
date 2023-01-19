import os
import random
import glob
import shutil

# read files "names/*.txt"
# select 1000 lines at random from each file
# store the selected lines in "names_1000/*.txt"

# read a file and return a list of lines
def read_file(file):
    with open(file, 'r', encoding="utf8") as f:
        return f.readlines()

# write a list of lines to a file
def write_file(file, lines):
    with open(file, 'w', encoding="utf8") as f:
        for line in lines:
            f.write(line)

# select n lines at random from a list of lines
def select_random_lines(lines, n):
    return random.sample(lines, n, counts=[int(10000/len(lines))+1]*len(lines),)

# select n lines at random and write them to a new file
def select_random_lines_from_files(files, n):
    for file in files:
        lines = read_file(file)
        selected_lines = select_random_lines(lines, n)
        write_file(file.replace('names', 'names_1000'), selected_lines)

if __name__=='__main__':
    files = glob.glob(os.path.join('names', '*.txt'))
    select_random_lines_from_files(files, 1000)
