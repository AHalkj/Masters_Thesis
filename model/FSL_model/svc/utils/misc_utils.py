import matplotlib.pyplot as plt

def change_fontsize():
    font_params = {
        'figure.titlesize': 20, 
        'axes.titlesize': 18,
        'figure.titleweight': 'bold',
        'axes.labelsize':14,
        'xtick.labelsize':14,
        'ytick.labelsize':14,
        'legend.fontsize': 16
    }
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(font_params)
    
def change_fontsize3x1():
    font_params = {
        'figure.titlesize': 24, 
        'axes.titlesize': 22,
        'figure.titleweight': 'bold',
        'axes.labelsize':18,
        'xtick.labelsize':18,
        'ytick.labelsize':18,
        'legend.fontsize': 20
    }
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(font_params)
    
simcorp_dark_green = '#012E26'
simcorp_light_green = '#3B9E87'
simcorp_light_yellow = '#E0AE22'
simcorp_dark_yellow = "#4A3306"