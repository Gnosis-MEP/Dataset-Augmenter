# Dataset Augmenter
Service responsible for creating/augmenting datasets with backgrounds and copy-and-paste OI foregrounds.
N is the target total number of images of each label.
B is the number of background samples. Create at least 3 background per sample: original, denoise( denoise_tv_chambolle(g_noised_image, multichannel=True)), and random_noise(land, mode='gaussian').
Z is number of combinations of OI Sum_i_K(K_C_i). (eg: 15 for K=4, K is number of OI).
Delta is the number of variations per combination of OI in Z. Default=7 , that is, 7 variations of cars, 7 variations of person, 7 variations car+person, for OI (car, person).


## Example case:

Targeting N ~= 600.
N == 630
B == 30 (FPS 30)
BV = N / B = 21 (background variations)
BR = BV - 2 = 19 . (random_noise)

K = 2 (person and car)
Z = Sum_i_K(K_C_i) == 3
Delta = roundUp(N / B / Z)  == 600 / 30 / 3 == 6.7 == 7.


Use this images as non-OI label.

Then create M images by random copy-and-paste of OI foregrounds using the Z combinations of OI foregrounds:

```
l = [1, 2 , 3, 4]
k = len(l)

comb_list = []
for i in range(1, k + 1):
    comb = list(combinations(l, i))
    print(comb)
    comb_list.extend(comb)
```

# Installation

## Configure .env
Copy the `example.env` file to `.env` with the correct information.

## Installing Dependencies

### Using pip
Load the environment variables from `.env` file using `source load_env.sh`.

To install from the `requirements.txt` file, run the following command:
```
$ pip install -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ python dataset_augmenter/simple_run.py HS-D-B-1-10S car
```
