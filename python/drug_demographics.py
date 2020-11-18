#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from matplotlib import pyplot as plt



# In[3]:


mydata = pd.read_csv('mydat_2.csv')


# In[4]:


mydata.info()


# In[6]:


mydata.shape


# In[8]:


#mydata = mydata.drop(columns = ['Unnamed: 0'])


# In[9]:


le = preprocessing.LabelEncoder()
for col in mydata.columns:
    mydata.loc[:, col] = le.fit_transform(mydata.loc[:, col])


# # Split the train and test set

# In[10]:


X_train, X_test = train_test_split(mydata, train_size = 0.75, stratify = mydata['shi'], random_state = 123)
y_train = X_train['shi']
y_test = X_test['shi']
X_train = X_train.drop(columns = ['shi'])
X_test = X_test.drop(columns = ['shi'])


# In[11]:


rforest = ensemble.RandomForestClassifier()
model = rforest.fit(X_train, y_train)


# # Feature Importance

# In[12]:


importance = model.feature_importances_
bars = X_train.columns
y_pos = range(len(bars))
plt.bar(y_pos, importance)
plt.xticks(y_pos, bars, rotation=90)
plt.show()


# # Train the model

# In[13]:


decisiontree = tree.DecisionTreeClassifier(random_state = 123, max_depth = 4)
model = decisiontree.fit(X_train, y_train)


# In[14]:


fig = plt.figure(figsize = (50, 50))
graph = tree.plot_tree(decisiontree, 
                   feature_names = X_train.columns,  
                   class_names = ['Yes', 'No'],
                   filled = True)


# In[15]:


pred = model.predict(X_test)
check = (pred == y_test)
sum(check)/len(check)


# # Freeze the model

# In[52]:


import platform
print(platform.python_version())


# In[18]:


import joblib


# In[19]:


filename = 'demodrug.pkl'
joblib.dump(decisiontree,filename)


# # Below steps involves AML Services

# # Import Azure ML SDK modules

# In[20]:


import os
import urllib
import shutil
import azureml
from azureml.core import Experiment
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

## Vicky-11122020 12:28 am - start
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id="b2ef2cdf-362a-48ec-8c76-321b322ed859", # tenant id
                                    service_principal_id="88226988-d739-4044-afb1-62220d8f507d", # clientId
                                    service_principal_password="s2MeaVDdTqlikvQlf3X0.WP7-T3WIZEDmF") # clientSecret
from azureml.core import Workspace

ws = Workspace.get(name="GAVS-ML-SPACE",
                   auth=sp,
                   subscription_id="758d9519-6a50-420c-a094-611f42144a79")

## Vicky-11122020 12:28 am - end

# # Load  and testing model

# In[21]:


filename = 'demodrug.pkl'
loaded_model = joblib.load(filename)
y = loaded_model.predict(X_test)
print(y)


# # Create the workspace

# In[23]:


# from azureml.core import Workspace

#ws = Workspace.create(name='POC-test2',
#                    subscription_id='d2a0e00c-84e7-40a7-96ce-4c730e4f85f7', 
#                    resource_group ='ResourceGp-ZS',
#                    create_resource_group=False,
#                    location='westus2' 
#                   )


# # If workspace is already created , directly run below code

# In[ ]:


# from azureml.core import Workspace

# subscription_id = 'd2a0e00c-84e7-40a7-96ce-4c730e4f85f7'
# resource_group  = 'ResourceGp-ZS'
# workspace_name  = 'POC-test'

# try:
#     ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
#     ws.write_config()
#     print('Library configuration succeeded')
# except:
#     print('Workspace not found')


# # Deploy
# Now that we have trained a set of models and identified the run containing the best model, we want to deploy the model for real time inferencing. The process of deploying a model involves registering a model in your workspace creating a scoring file containing init and run methods creating an environment dependency file describing packages necessary for your scoring file creating a docker image containing a properly described environment, your model, and your scoring file deploying that docker image as a web service

# # Register a model
# We have already identified which run contains the "best model" by our evaluation criteria. Each run has a file structure associated with it that contains various files collected during the run. Since a run can have many outputs we need to tell AML which file from those outputs represents the model that we want to use for our deployment. We can use the run.get_file_names() method to list the files associated with the run, and then use the run.register_model() method to place the model in the workspace's model registry.
# When using run.register_model() we supply a model_name that is meaningful for our scenario and the model_path of the model relative to the run. In this case, the model path is what is returned from run.get_file_names()

# In[25]:


from azureml.core.model import Model
model = Model.register(model_path = "demodrug.pkl",
                       model_name = "demodrug",
                       description = "SHI Decision Trees",
                       workspace = ws)
print("Model location : ")
print(model.url)


# # Create a container image for our model

# # Create a scoring file
# Since your model file can essentially be anything you want it to be, you need to supply a scoring script that can load your model and then apply the model to new data. This script is your 'scoring file'. This scoring file is a python program containing, at a minimum, two methods init() and run(). The init() method is called once when your deployment is started so you can load your model and any other required objects. This method uses the get_model_path function to locate the registered model inside the docker container. The run() method is called interactively when the web service is called with one or more data samples to predict.

# In[26]:


#This script will provide a prediction of classification for new observations that API will pull out


# In[27]:


#get_ipython().run_cell_magic('writefile', 'score.py', "\nimport json\nimport sys\nimport joblib\n\nfrom azureml.core.model import Model\nimport numpy as np\n\ndef init():\n\n    global path\n    model_path = Model.get_model_path('demodrug')\n    model = joblib.load(model_path)\n\ndef run(raw_data):\n    try:\n        data = json.loads(raw_data)['data']\n        data = numpy.array(data)\n        result  = model.predict(data)\n        return result.tolist()\n    except Exception as e:\n        result = str(e)\n        return error")

# # Describe your environment
# Each modelling process may require a unique set of packages. Therefore we need to create a dependency file providing instructions to AML on how to contstruct a docker image that can support the models and any other objects required for inferencing. In the following cell, we create a environment dependency file, myenv.yml that specifies which libraries are needed by the scoring script. You can create this file manually, or use the CondaDependencies class to create it for you.
# 
# Next we use this environment file to describe the docker container that we need to create in order to deploy our model. This container is created using our environment description and includes our scoring script.
# 

# In[28]:


from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.environment import Environment

env = Environment(name= "env")

myenv = CondaDependencies()
myenv.add_pip_package("numpy")
myenv.add_pip_package("azureml-core")
myenv.add_pip_package("sklearn")

# Adds dependencies to PythonSection of myenv
env.python.conda_dependencies = myenv

print(myenv.serialize_to_string())

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


# # Create an image configuration

# # Deploy your webservice
# The final step to deploying your webservice is to call WebService.deploy_from_model(). This function uses the deployment and image configurations created above to perform the following:
# Build a docker image Deploy to the docker image to an Azure Container Instance Copy your model files to the Azure Container Instance Call the init() function in your scoring file Provide an HTTP endpoint for scoring calls The deploy_from_model method requires the following parameters
# workspace - the workspace containing the service name - a unique named used to identify the service in the workspace models - an array of models to be deployed into the container image_config - a configuration object describing the image environment deployment_config - a configuration object describing the compute type Note: The web service creation can take several minutes.

# # Deploy image as web service on Azure Container Instance
# Note that the service creation can take few minutes.

# In[32]:


from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice

# Register the model to deploy
#model = run.register_model(model_name = "mymodel", model_path = "outputs/model.pkl")

# Combine scoring script & environment in Inference configuration
inference_config = InferenceConfig(entry_script="python/score.py", environment = env)

# Set deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

# check if exists - update or create 
DEPLOYMENT_SERVICE_NAME = "get-dgraphics-restlink"
webservices = ws.webservices.keys()
if DEPLOYMENT_SERVICE_NAME not in webservices:
     # Define the model, inference, & deployment configuration and web service name and location to deploy
     service = Model.deploy(
     workspace = ws,
     name = DEPLOYMENT_SERVICE_NAME,
     models = [model],
     inference_config = inference_config,
     deployment_config = deployment_config)
     
     service.wait_for_deployment(show_output=True)
     print(service.state)
     print(service.scoring_uri)
     print(service.get_logs())   

else:
    service = Webservice(
                  name=DEPLOYMENT_SERVICE_NAME,
                  workspace=ws
    )
    service.update(models=[model], inference_config=inference_config)
    print(service.state)
    print(service.scoring_uri)
    print(service.get_logs())


# In[33]:


#service.wait_for_deployment(show_output = True)


# In[34]:


#service_name = 'ddshi1'
#service = Webservice(name = service_name, workspace = ws)
#print(service.get_logs())
#print(service.state)


# In[35]:


#print(service.scoring_uri)

