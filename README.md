# chicken-disease-classification
End-to-end project CNN based project, also demonstrating the MLOps practices, along with deployment.


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity  # to create some return type/classes
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml


# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/satyamsc0/chicken-disease-classification
```
### STEP 01- Create a virtual environment after opening the repository

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag




# AZURE-CICD-Deployment-with-Github-Actions

## Save pass:
EZKH5yytiVnJ3h+eI3qhhzf9q1vNwEi6+q+WGdd+ACRCZ7J


## Run from terminal:

docker build -t chickenreg.azurecr.io/chicken:latest .

docker login chickenreg.azurecr.io

docker push chickenreg.azurecr.io/chicken:latest


## Deployment Steps:

1. Build the Docker image of the Source Code
2. Push the Docker image to Container Registry
3. Launch the Web App Server in Azure 
4. Pull the Docker image from the container registry to Web App server and run 