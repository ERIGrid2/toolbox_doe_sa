import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas
import datetime
from loguru import logger


FIGURES_FOLDER = 'figures'
IMAGE_FORMAT = 'png'


def analyse_mrsq_demo_pv():
    uncertain = 'results\\mrsq_pv.hdf5'

    with h5py.File(uncertain, "r") as f:
        sun_dni_max = f['Series']['OAM-0.ItvlOut_0']['DNI_maxA'][:]
        sun_dni_min = f['Series']['OAM-0.ItvlOut_0']['DNI_minA'][:]
        pv_power_max = f['Series']['OAM-1.ItvlOut_0']['P_maxA'][:]
        pv_power_min = f['Series']['OAM-1.ItvlOut_0']['P_minA'][:]

    fig, ax = plt.subplots()
    ax.plot(sun_dni_max, color='red', label='DNI max')
    ax.plot(sun_dni_min, color='blue', label='DNI min')
    fig.suptitle(f'DNI Uncertainty')
    ax.set_ylabel(f'DNI [W/M²]')
    ax.set_xlabel(f'Time')
    fig.savefig(f'{FIGURES_FOLDER}\\mrsq_pv\\pv_example_2.{IMAGE_FORMAT}', dpi=300, format=IMAGE_FORMAT)

    fig, ax = plt.subplots()
    ax.plot(pv_power_max, color='blue', label='P max')
    ax.plot(pv_power_min, color='red', label='P min')
    fig.suptitle(f'Power Uncertainty')
    ax.set_ylabel(f'P')
    ax.set_xlabel(f'Time')
    fig.savefig(f'{FIGURES_FOLDER}\\mrsq_pv\\pv_example_3.{IMAGE_FORMAT}', dpi=300, format=IMAGE_FORMAT)


def analyse_mrsq_demo_scenario():
    uncertain = 'results\\mrsq_scenario.hdf5'
    figures_template = f'{FIGURES_FOLDER}\\mrsq_scenario\\uq_example'
    with h5py.File(uncertain, "r") as f:
        series = list(f['Series'])
        sun_dni_max = f['Series']['OAM-0.ItvlOut_0']['DNI_maxA'][:]
        sun_dni_min = f['Series']['OAM-0.ItvlOut_0']['DNI_minA'][:]

        fig, ax = plt.subplots()
        ax.plot(sun_dni_max, color='blue', label='DNI max')
        ax.plot(sun_dni_min, color='red', label='DNI min')
        ax.set_title(f'DNI')
        ax.legend()
        fig.savefig(f'{figures_template}_dni.{IMAGE_FORMAT}', format="png")
        #
        for i in range(5):
            house_p = f['Series'][f'CSV-1.House_{i}']['P'][:]

            fig, ax = plt.subplots()
            ax.plot(house_p, color='black', linestyle='solid', label='House P')
            fig.suptitle(f'House {i}')
            ax.set_ylabel(f'Power P')
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_house_{i}.{IMAGE_FORMAT}', format="png")

        for i in range(5):
            pv_avg_a = f['Series'][f'OAM-1.PboxOut_{i}']['P_aveA'][:]
            pv_q05_a = f['Series'][f'OAM-1.PboxOut_{i}']['P_q05A'][:]
            pv_q15_a = f['Series'][f'OAM-1.PboxOut_{i}']['P_q15A'][:]
            pv_q86_a = f['Series'][f'OAM-1.PboxOut_{i}']['P_q84A'][:]
            pv_q95_a = f['Series'][f'OAM-1.PboxOut_{i}']['P_q95A'][:]

            fig, ax = plt.subplots()
            ax.plot(pv_avg_a, color='black', linestyle='solid', label='PV avg A')
            ax.plot(pv_q05_a, color='green', linestyle='dashed',  label='PV q05 A')
            ax.plot(pv_q15_a, color='blue', linestyle='dotted',  label='PV q15 A')
            ax.plot(pv_q86_a, color='blue', linestyle='dotted',  label='PV q86 A')
            ax.plot(pv_q95_a, color='green', linestyle='dashed',  label='PV q95 A')
            fig.suptitle(f'PV {i}')
            ax.set_ylabel(f'Power P')
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_pv_{i}.{IMAGE_FORMAT}', format="png")

        for i in range(6):
            branch_avg_a = f['Series'][f'OAM-2.PboxOut_{i}']['Vm_aveA'][:]
            branch_q05_a = f['Series'][f'OAM-2.PboxOut_{i}']['Vm_q05A'][:]
            branch_q15_a = f['Series'][f'OAM-2.PboxOut_{i}']['Vm_q15A'][:]
            branch_q86_a = f['Series'][f'OAM-2.PboxOut_{i}']['Vm_q84A'][:]
            branch_q95_a = f['Series'][f'OAM-2.PboxOut_{i}']['Vm_q95A'][:]

            fig, ax = plt.subplots()
            ax.plot(branch_avg_a, color='black', linestyle='solid', label='Branch avg A')
            ax.plot(branch_q05_a, color='green', linestyle='dashed',  label='Branch q05 A')
            ax.plot(branch_q15_a, color='blue', linestyle='dotted',  label='Branch q15 A')
            ax.plot(branch_q86_a, color='blue', linestyle='dotted',  label='Branch q86 A')
            ax.plot(branch_q95_a, color='green', linestyle='dashed',  label='Branch q95 A')
            fig.suptitle(f'PQBus {i}')
            ax.set_ylabel(f'Voltage Vm')
            ax.set_ylim([229, 233.5])
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_bus_{i}.{IMAGE_FORMAT}', dpi=300, format=IMAGE_FORMAT)


def analyse_mrsq_extended_use_case():
    uncertain = 'results\\mrsq_scenario2.hdf5'
    figures_template = f'{FIGURES_FOLDER}\\mrsq_extended\\mrsq_scenario2'
    with h5py.File(uncertain, "r") as f:
        sun_dni_max = f['Series']['OAM-0.ItvlOut_0']['DNI_maxA'][:]
        sun_dni_min = f['Series']['OAM-0.ItvlOut_0']['DNI_minA'][:]

        fig, ax = plt.subplots()
        ax.plot(sun_dni_max, color='blue', label='DNI max')
        ax.plot(sun_dni_min, color='red', label='DNI min')
        ax.set_title(f'DNI')
        ax.legend()
        fig.savefig(f'{figures_template}_dni.{IMAGE_FORMAT}', format="png")
        #
        for i in range(5):
            house_avg = f['Series'][f'OAM-1.DstrOut_{i}']['P_aveA'][:]
            house_q05 = f['Series'][f'OAM-1.DstrOut_{i}']['P_q05A'][:]
            house_q15 = f['Series'][f'OAM-1.DstrOut_{i}']['P_q15A'][:]
            house_q86 = f['Series'][f'OAM-1.DstrOut_{i}']['P_q84A'][:]
            house_q95 = f['Series'][f'OAM-1.DstrOut_{i}']['P_q95A'][:]

            fig, ax = plt.subplots()
            ax.plot(house_avg, color='black', linestyle='solid', label='House avg')
            ax.plot(house_q05, color='red', linestyle='dashed',  label='House q05')
            ax.plot(house_q15, color='blue', linestyle='dotted',  label='House q15')
            ax.plot(house_q86, color='blue', linestyle='dotted',  label='House q86')
            ax.plot(house_q95, color='red', linestyle='dashed',  label='House q95')
            fig.suptitle(f'House {i}')
            ax.set_ylabel(f'Power P')
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_house_{i}.{IMAGE_FORMAT}', format="png")

        for i in range(5):
            pv_avg_a = f['Series'][f'OAM-2.PboxOut_{i}']['P_aveA'][:]
            pv_q05_a = f['Series'][f'OAM-2.PboxOut_{i}']['P_q05A'][:]
            pv_q15_a = f['Series'][f'OAM-2.PboxOut_{i}']['P_q15A'][:]
            pv_q86_a = f['Series'][f'OAM-2.PboxOut_{i}']['P_q84A'][:]
            pv_q95_a = f['Series'][f'OAM-2.PboxOut_{i}']['P_q95A'][:]
            pv_avg_b = f['Series'][f'OAM-2.PboxOut_{i}']['P_aveB'][:]
            pv_q05_b = f['Series'][f'OAM-2.PboxOut_{i}']['P_q05B'][:]
            pv_q15_b = f['Series'][f'OAM-2.PboxOut_{i}']['P_q15B'][:]
            pv_q86_b = f['Series'][f'OAM-2.PboxOut_{i}']['P_q84B'][:]
            pv_q95_b = f['Series'][f'OAM-2.PboxOut_{i}']['P_q95B'][:]

            fig, ax = plt.subplots()
            ax.plot(pv_avg_a, color='green', linestyle='solid', label='PV avg A')
            ax.plot(pv_q05_a, color='green', linestyle='dashed',  label='PV q05 A')
            ax.plot(pv_q15_a, color='green', linestyle='dotted',  label='PV q15 A')
            ax.plot(pv_q86_a, color='green', linestyle='dotted',  label='PV q86 A')
            ax.plot(pv_q95_a, color='green', linestyle='dashed',  label='PV q95 A')
            ax.plot(pv_avg_b, color='blue', linestyle='solid', label='PV avg B')
            ax.plot(pv_q05_b, color='blue', linestyle='dashed',  label='PV q05 B')
            ax.plot(pv_q15_b, color='blue', linestyle='dotted',  label='PV q15 B')
            ax.plot(pv_q86_b, color='blue', linestyle='dotted',  label='PV q86 B')
            ax.plot(pv_q95_b, color='blue', linestyle='dashed',  label='PV q95 B')
            fig.suptitle(f'PV {i}')
            ax.set_ylabel(f'Power P')
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_pv_{i}.{IMAGE_FORMAT}', format="png")

        for i in range(5):
            branch_avg_a = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_aveA'][:]
            branch_q05_a = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q05A'][:]
            branch_q15_a = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q15A'][:]
            branch_q86_a = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q84A'][:]
            branch_q95_a = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q95A'][:]
            branch_avg_b = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_aveB'][:]
            branch_q05_b = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q05B'][:]
            branch_q15_b = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q15B'][:]
            branch_q86_b = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q84B'][:]
            branch_q95_b = f['Series'][f'OAM-3.PboxOut_{i}']['Vm_q95B'][:]

            fig, ax = plt.subplots()
            ax.plot(branch_avg_a, color='green', linestyle='solid', label='Branch avg A')
            ax.plot(branch_q05_a, color='green', linestyle='dashed',  label='Branch q05 A')
            ax.plot(branch_q15_a, color='green', linestyle='dotted',  label='Branch q15 A')
            ax.plot(branch_q86_a, color='green', linestyle='dotted',  label='Branch q86 A')
            ax.plot(branch_q95_a, color='green', linestyle='dashed',  label='Branch q95 A')
            ax.plot(branch_avg_b, color='blue', linestyle='solid', label='Branch avg B')
            ax.plot(branch_q05_b, color='blue', linestyle='dashed',  label='Branch q05 B')
            ax.plot(branch_q15_b, color='blue', linestyle='dotted',  label='Branch q15 B')
            ax.plot(branch_q86_b, color='blue', linestyle='dotted',  label='Branch q86 B')
            ax.plot(branch_q95_b, color='blue', linestyle='dashed',  label='Branch q95 B')
            fig.suptitle(f'PQBus {i}')
            ax.set_ylabel(f'Voltage Vm')
            ax.set_ylim([229, 233.5])
            ax.set_xlabel(f'Simulation steps')
            ax.legend()
            fig.savefig(f'{figures_template}_bus_{i}.{IMAGE_FORMAT}', dpi=300, format=IMAGE_FORMAT)


if __name__ == '__main__':
    analyse_mrsq_demo_pv()
    analyse_mrsq_demo_scenario()
    analyse_mrsq_extended_use_case()

