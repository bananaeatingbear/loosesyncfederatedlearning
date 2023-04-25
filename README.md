## Effective passive membership inference attacks in federated learning against overparameterized models

This is the code repo for the ICLR 2023 paper "Effective passive membership inference attacks in federated learning against overparameterized models". This repo is tested using the following libraries: 

- Python 3.7.5
- Pytorch 1.12.1
- Scipy 1.4.1
- Scikit-learn 1.0.1
- Numpy 1.20.3

------

### Code

Data subdirectory contains code to process the data into numpy array.

Model subdirectory contains code for implementation of different models.

Utils subdirectory contains code for some utility functions.

Different attacks are implemented in blackbox_attack.py.

Our propose cosine attack and gradient-diff attack are implemented in fed_attack_epoch.py. Some other baseline attacks are implemented in fed_attack.py.

fed_ttack_exp.py contains code for model training and membership inferecen attack evaluation. There is a path name that should be configured to your local enviroment. I use './expdata/' for my own environment, please remember to change this path. Most parameters can be figured out in this python file by changing the default value or use fed_attack_exp.sh to specify.

fed_attack_exp.sh is used to run the experiment and there are a few arguments that need to be figured out. I included some script that I use. Hope this would help.

-----

### Citation

If you use this code, please cite the following paper:

```
@inproceedings{lieffective,
  title={Effective passive membership inference attacks in federated learning against overparameterized models},
  author={Li, Jiacheng and Li, Ninghui and Ribeiro, Bruno},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```

-----

### Notes

Notes for datasets used in this paper: due to the file size limit, we are not able to provide you the datasets we use but basically we process each dataset into 4 different numpy arrays: train_data, train_labels, test_data and test_labels. After this, we load these 4 numpy arrays and process the data into disjoint parts for training/testing/membership inference evaluation. I can share the data upon request.


If you have any other questions, please feel free to email me at li2829@purdue.edu.

