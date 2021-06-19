from matplotlib import pyplot as plt
import json
import numpy as np


def load_array_from_tb_json(path):
    if '.json' in path:
        with open(path) as json_file:
            data = json.load(json_file)
            data = np.asarray(data)[:, 1:]
    else:
        with open(path) as txt_file:
            lines = [l.strip() for l in txt_file.readlines() if l.strip() != ""]
            data = []
            for l in lines:
                epoch, fid = l.split(':')
                epoch = int(epoch)
                fid = float(fid)
                data.append([epoch, fid])
            data = np.asarray(data)
    return data


def plot_array(a, label=None):
    x = a[:, 0]
    y = a[:, 1]
    plt.plot(x, y, label=label)


def show_plot(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def all_plotting(dataset_names, json_path, *args, **kwargs):
    for ds_name in dataset_names:
        plot_array(load_array_from_tb_json(json_path.format(ds_name)), ds_name)
    show_plot(*args, **kwargs)


# GRAF
def plot_graf_plots():
    carla_path_name = 'carla'
    sncars_path_name = 'shapenetcars'
    snships_path_name = 'shapenetships'
    fid_path = 'new_plan_result/sym_loss/run-sncar_128_{}_monitoring-tag-validation_fid.json'
    # kid_path = 'new_plan_result/run-{}_128_new_plan_monitoring-tag-validation_kid.json'

    all_plotting(
        ["baseline", "no_mirror_baseline", "sym_loss"],
        # ['shapenetcars_varying_distance'],
        fid_path,
        'Frechet Inception Distance for GRAF',
        'Epoch',
        'FID'
    )
    #
    # all_plotting(
    #     [carla_path_name, sncars_path_name, snships_path_name],
    #     kid_path,
    #     'Kernel Inception Distance for GRAF',
    #     'Epoch',
    #     'KID'
    # )



# Pi-GAN
def plot_pi_gan_plots():
    carla_path_name = 'carla'
    sncars_path_name = 'shapenetcars'
    snships_path_name = 'shapenetships'
    fid_path = 'new_plan_result/sgan_{}_fid.txt'

    all_plotting(
        [carla_path_name, sncars_path_name, snships_path_name],
        fid_path,
        'Frechet Inception Distance for Pi-GAN',
        'Epoch',
        'FID'
    )


def main():
    plot_graf_plots()
    # plot_pi_gan_plots()


if __name__ == '__main__':
    import os
    from getpass import getpass

    # password = getpass()
    # if password:
    #     peregrine_adress = 'sshpass -p "{}" scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_{}/fid.txt new_plan_result/sgan_{}_fid.txt'
    #     peregrine_names = ['shapenetcars', 'shapenetships', 'cars']
    #     local_names = ['shapenetcars', 'shapenetships', 'carla']
    #     for p, l in zip(peregrine_names, local_names):
    #         os.system(peregrine_adress.format(password, p, l))

    main()


# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_shapenetcars/fid.txt new_plan_result/sgan_shapenetcars_fid.txt
# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_shapenetships/fid.txt new_plan_result/sgan_shapenetships_fid.txt
# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_cars/fid.txt new_plan_result/sgan_carla_fid.txt