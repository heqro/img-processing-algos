for index in range(1, 11):
    path = f'synth_img_{index}/results_log/raw.txt'
    output = f'synth_img_{index}/results_log/parsed_raw.csv'
    with open(path, 'r') as raw_file:
        with open(output, 'a') as raw_output_file:
            raw_output_file.writelines('width,height,noise_std,n_it,t_diff,T_exec,mu\n')
            for line in raw_file:  # output as csv
                cols = line.strip('\n').split(sep=' ')
                width, height, std, dt, n_it, t_exec, mu = cols[1], cols[2], cols[4], 1e-2, cols[14], cols[6], cols[8]
                t_diff = dt * int(n_it)
                output_line = f'{width},{height},{std},{n_it},{t_diff},{t_exec},{mu}\n'
                # print(output_line,end='')  # dry-run
                raw_output_file.writelines(output_line)
