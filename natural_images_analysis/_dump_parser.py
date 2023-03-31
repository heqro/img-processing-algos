p = '1'
with open(f'all_analysis_p{p}_copy.csv', mode='a') as output:
    for index in range(101):
        with open(f'img_{index}/analysis_p{p}_copy.csv', mode='r') as input_dump:
            i = 0
            for line in input_dump:
                if i == 0:
                    i += 1
                    continue
                output.writelines(line)



# import pandas as pd
# for index in range(101):
#     df = pd.read_csv(f'img_{index}/analysis.csv',header=0)
#     res = df.drop('avg_mass_loss',axis=1)
#     res.to_csv(f'img_{index}/analysis.csv')


# for index in range(101):
#     filename = f'img_{index}/analysis.csv'
#     with open(filename, "r") as file:
#         # Read the contents of the file into a list of lines
#         lines = file.readlines()
#
#         # Modify the first line
#         lines[0] = lines[0][1:]
#
#         # Modify the other lines
#         for i in range(1, len(lines)):
#             lines[i] = lines[i][2:]
#
#     with open(filename, "w") as file:
#         # Write the modified contents back to the same file
#         file.writelines(lines)
