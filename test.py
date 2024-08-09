# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from phase_generator import *
from astropy.io import fits


hdul = fits.open("./devs/HarmoniLTAO_JQ2_undersampled.fits")

primary_hdu = hdul[0]
psd = hdul[4].data[1, :, :]

# Parameters
D = 39  # Size of the phase screen [m]
r0 = 0.1  # Fried parameter [m]
L0 = 25.0  # Outer scale [m]

dx = r0 / 3.0  # Spatial sampling interval [m/pixel], make sure r0 is Nyquist sampled
dt = 0.001  # Time step [s/step]

wind_speed = 40  # [m/s]
wind_direction = 15  # [degree]
boiling_factor = 500  # [a.u], need to figure them out


for batch_size in [5, 10, 50, 100, 500, 1000, 5000]:

    screen_generator = CascadedPhaseGenerator(
        D, dx, dt, batch_size=batch_size, n_cascades=1
    )
    screen_generator.AddLayer(
        1.0, r0, L0, wind_speed, wind_direction, boiling_factor, PSDs=psd
    )
    # screen_generator.AddLayer(0.5, r0/2., L0, wind_speed*2, wind_direction*4, boiling_factor)

    # %%

    print(f"GPU_flag: {GPU_flag}")

    if GPU_flag:
        start = cp.cuda.Event()
        end = cp.cuda.Event()

    total_time = 0
    screens_cascade = []

    t0 = time.time()
    N = 50000

    for i in tqdm(range(N)):
        if GPU_flag:
            start.record()
        else:
            start = time.time()

        # screens_cascade.append(screen_generator.GetScreenByTimestep(i))
        screen_generator.GetScreenByTimestep(i)

        if GPU_flag:
            end.record()
            end.synchronize()
            total_time += cp.cuda.get_elapsed_time(start, end)  # Time in [ms]
        else:
            end = time.time()
            total_time += (end - start) * 1000

    print(
        f"Batch size: {batch_size} / Time per screen: {np.round(total_time/N, 3)} [ms]"
    )

# %%
