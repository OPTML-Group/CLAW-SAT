import matplotlib.pyplot as plt
import os
import json

def plot_loss(all_files):
    files = os.listdir(all_files)
    files.sort() # sort by epoch number
    epoch = []
    train_loss = []
    dev_loss = []
    for f in files:
        if '.json' not in f:
            continue
        with open(os.path.join(all_files, f), 'r') as f_in:
            data = json.load(f_in)
        for k, item in data.items():
            epoch.append(int(k.replace('epoch ', '')))
            train_loss.append(data[k]['train_loss'])
            dev_loss.append(data[k]['dev_loss'])
    train_plot = plt.plot(train_loss)
    dev_plot = plt.plot(dev_loss)
    plt.xticks(range(len(epoch)), range(1, len(epoch)+1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.savefig(os.path.join(all_files,'loss.png'))
    # print(epoch)
    # print(train_loss)
    # print(dev_loss)

if __name__ == '__main__':
    expt_name = 'v2-3-z_rand_1-pgd_3-no-transforms.Replace-py150-normal'
    plots_dir = os.path.join('plots', expt_name)
    plot_loss(plots_dir)

