path = "2-5-6-7.txt"

with open(path, mode='r') as file:
    for line in file:
        cols = line.split(' ')
        index = cols[3]
        with open(f'synth_img_{index}/results_log/raw.txt', mode='a') as output_file:
            output_file.writelines(line)
