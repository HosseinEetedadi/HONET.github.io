# simulator-2021-nca

## Introduction 

This simulator has been described and used in the paper "*Leveraging Reinforcement Learning for online scheduling of real-time tasks in the Edge/Fog-to-Cloud computing continuum*" (https://dx.doi.org/10.1109/NCA53618.2021.9685413).

You are free to use it and improve it but if do it please **cite** our work and **redistribute** the code by using the same licence.

See the paper (https://gpm.name/publications/2021ProiettiMattiaLeveraging/) for the description of what the simulator implements.


Thanks!

## Citation

Proietti Mattia, G., & Beraldi, R. (2021). Leveraging Reinforcement Learning for online scheduling of real-time tasks in the Edge/Fog-to-Cloud computing continuum. 2021 IEEE 20th International Symposium on Network Computing and Applications (NCA), 1–9. https://doi.org/10.1109/NCA53618.2021.9685413

```bibtex
@inproceedings{2021ProiettiMattiaLeveraging,
  author = {{Proietti Mattia}, Gabriele and Beraldi, Roberto},
  booktitle = {2021 IEEE 20th International Symposium on Network Computing and Applications (NCA)},
  title = {Leveraging Reinforcement Learning for online scheduling of real-time tasks in the Edge/Fog-to-Cloud computing continuum},
  year = {2021},
  volume = {},
  number = {},
  pages = {1-9},
  doi = {10.1109/NCA53618.2021.9685413}
}
```

## Instructions

The code is written with modularity in mind. The filename convention is the following:
- `run_*.py` - run the simulations
- `node.py` - implements a fog/edge node
- `log.py` - logging utilities
- `service_*.py` - implements the services used during the simulations
- `plot_*.py` - plot results
- `test_*.py` - scripts for testing functions used in the simulator
