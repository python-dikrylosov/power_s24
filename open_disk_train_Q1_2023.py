import os
import pandas as pd
import matplotlib.pylab as plt

path = 'data_Q1_2023/'
filenames = []
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        filenames.append(filename)

for fname in filenames:
    df = pd.read_csv(os.path.join(path, fname))
    print(f"Read failure column from file {fname}")
    print(df['failure'])
    for i in range(df.shape[0]):
        if df['failure'][i] > 0 :
                print([df['serial_number'][i],df['failure'][i]])
    plt.title(fname)
    plt.plot(df['failure'])
    plt.show()

    # Здесь выполняете операции с данными

# Или, если нужно прочитать данные сразу за все дни:
dfs = [pd.read_csv(os.path.join(path, fname)) for fname in filenames]
combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df)
