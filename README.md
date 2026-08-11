This is the code used in the paper "Parametrization of a steady-state analytical model to estimate offshore wind farm wakes in the Baltic Sea". You will need access to the [Jupyter Notebook](https://jupyter.org/) software and a console from which you can run a python environment (I was using version 3.14.3).

Each file has a key purpose and an implicit order :

- **"CropToRegion.py"** (executed from a console) crops the input dataset and regrids it to a regular latitude/longitude grid of the specified region. The data needs to be on NetCDF (.nc) format. The name of the dataset's variable can be adjusted although the regridding process should always be applied to each component of the wind speed instead of the norm value.
- **"Correlation10mWinds.ipynb"** contains the analysis for the 10-meter winds. It goes through all the reference files in January, February, March and April to revetrieve the variables of interest and then fit a predetermined form on the data. The file can be launched in order in Jupyter Notebook.
- **"RefModelEnergyDeficit.ipynb"** contains the first part of the second correlation (rotor diameter). The analysis is ran on the reference model.
- **"PyWakeEnergyDeficit.ipynb"** contains the second part of the second correlation. The analysis is ran on the PyWake software, particularly the wake deficit model.
- **"pywake_DMI_area.py"** (executed from a console) contains the code running simulations with PyWake. Both correlations for the 10-meter winds and the rotor diameter are applied. The file needs two arguments when executed: a wind farm name and a month (useful when dealing with datasets from several wind farms and several months). It will then look in the data_folder for files of this wind farm and this month.
- **"pywake_original.py"** (executed from a console) contains the code running simulations with PyWake. Only the correlation for the 10-meter winds is applied. The file also needs two arguments when executed: a wind farm name and a month.
- **"pywake_SHEAR.py"** (executed from a console) contains the code to divide simulations in low, mid and high vertical shear conditions. The code can be adjusted to compute either the output with only the 10-meter winds correlation or both.
- **"ShearComparison.ipynb"** contains the code to calculate PyWake's output in the three vertical shear conditions using the Power and MOST (Monin-Obukhov Similarity Theory) Shear objects embedded in PyWake. It is used to plot the two shear-dependent figures shown in the paper. It also contains code to average the results over each shear condition.
- **"HAMOWIOWF_Avg_Analysis.ipynb"** contains code to average PyWake's output on a daily and monthly basis (along with a total average). An example code to plot the data is proposed.  

In general in the code :
- **"noWF"** (as opposed to "WF") refers to datasets without the influence of Wind Farms (and respectively with the influence of Wind Farms).
- The expressions **"h"**, **"hamo"** or **"hAVG"** are associated with the reference models (however "h" on its own is height).
- The expressions **"p"**, **"py"** or **"pAVG"** are associated with the output of PyWake.
- "data_folder" is the folder from which you draw the datasets.
- "work_dir" is the folder in which some complementary files are found, such as the regridding weights "h_to_regular_precise.nc" or the locations of the turbines in the "turbines_offshore.csv" table.
- "output_dir" is the folder in which the output will go.

The folder space follows this kind of architecture :
- data/
  - KriegersFlak/ <p align="right">_(wind farm name example)_</p>
    - PyWake/ <p align="right">_(PyWake's output)_</p>
      - original/ <p align="right">_(PyWake's output but with only the first correlation on 10-meter winds)_</p>
      - _\[...\]_
    - withOWF/ <p align="right">_(reference model with Offshore Wind Farms)_</p>
      - _\[...\]_
    - withoutOWF/ <p align="right">_(reference model without Offshore Wind Farms)_</p>
      - _\[...\]_
  - \[Name of the wind farm\]/
    - PyWake/
    - withOWF/
    - withoutOWF/
  - Shear/
    - low/
    - mid/
    - high/

The "Shear" folder contains the reference models' cropped datasets along with the outputs for PyWake for all of the data, divided in three vertical shear conditions defined in the paper. As they refer to the value of the ratio between wind speeds at 10m and 100m, the "low" directory actually refers to high shear (and vice-versa).
