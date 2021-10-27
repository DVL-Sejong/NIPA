# NIPA

### Dataset

Dataset for this repository can be downloaded [here](https://github.com/DVL-Sejong/COVID_DataProcessor). You must download data and preprocess the data for the model. Dataset for the model should be under `\dataset\country_name`.



### NIPA

```
$ git clone https://github.com/DVL-Sejong/NIPA.git
$ cd NIPA
$ python main.py
```

- Arguments
  - country:  Italy, India, US, China are available
  - standardization: if True, standardize dataset while inference network. We recommend False
  - x_frames: Number of x frames for generating dataset
  - y_frames: Number of y frames for generating dataset
- Results are saved under `results\country_name\` and `settiings\`



### Citation

NIPA code is based on this paper:

```
@article{prasse2020network,
  title={Network-inference-based prediction of the COVID-19 epidemic outbreak in the Chinese province Hubei},
  author={Prasse, Bastian and Achterberg, Massimo A and Ma, Long and Van Mieghem, Piet},
  journal={Applied Network Science},
  volume={5},
  number={1},
  pages={1-11},
  year={2020},
  publisher={SpringerOpen}
}
```

