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

# Commands Stream
## Inputs
...

## Outputs
...

# Data Stream
## inputs
...

## Outputs
...

# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside it replace `SIT_PYPI_USER` and `SIT_PYPI_PASS` with the correct information.

## Installing Dependencies

### Using pipenv
Run `$ pipenv shell` to create a python virtualenv and load the .env into the environment variables in the shell.

Then run: `$ pipenv install` to install all packages, or `$ pipenv install -d` to also install the packages that help during development, eg: ipython.
This runs the installation using **pip** under the hood, but also handle the cross dependency issues between packages and checks the packages MD5s for security mesure.


### Using pip
To install using pip directly, one needs to use the `--extra-index-url` when running the `pip install` command, in order for to be able to use our private Pypi repository.

Load the environment variables from `.env` file using `source load_env.sh`.

To install from the `requirements.txt` file, run the following command:
```
$ pip install --extra-index-url https://${SIT_PYPI_USER}:${SIT_PYPI_PASS}@sit-pypi.herokuapp.com/simple -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ ./dataset_augmenter/run.py
```

# Testing
Run the script `run_tests.sh`, it will run all tests defined in the **tests** directory.

Also, there's a python script at `./dataset_augmenter/send_msgs_test.py` to do some simple manual testing, by sending msgs to the service stream key.


# Docker
## Manual Build (not recommended)
Build the docker image using: `docker-compose build`

**ps**: It's required to have the .env variables loaded into the shell so that the container can build properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

## Run
Use `docker-compose run --rm service` to run the docker image


## Gitlab CI auto-build and tests

This is automatically enabled for this project (using the `.gitlab-ci.yml` present in this project root folder).

By default it will build the Dockerfile with every commit sent to the origin repository and tag it as 'dev'.

Afterwards, it will use this newly builty image to run the tests using the `./run_tests.sh` script.

But in order to make the automatic docker image build work, you'll need to set the `SIT_PYPI_USER` and `SIT_PYPI_PASS` variables in the Gitlab CI setting page: [Dataset Augmenter CI Setting Page](https://gitlab.insight-centre.org/sit/mps/felipe-phd/dataset-augmenter/settings/ci_cd). (Or make sure the project is set under a Gitlab group that has this setup for all projects in that group).

And, in order to make the automatic tests work, you should also set the rest of the environement variables required by your service, in the this projects `.gitlab-ci.yml` file, in the `variables` section. But don't add sensitive information to this file, such as passwords, this should be set through the Gitlab CI settings page, just like the `SIT_PYPI_USER`.

## Benchmark Tests
To run the benchmark tests one needs to manually start the Benchmark stage in the CI pipeline, it shoud be enabled after the tests stage is done. Only by passing the benchmark tests shoud the image be tagged with 'latest', to show that it is a stable docker image.