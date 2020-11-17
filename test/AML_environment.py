from azureml.core import Workspace
import os, json, sys
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication

print("SDK Version:", azureml.core.VERSION)
# print('current dir is ' +os.curdir)

workspace_name = 'GAVS-ML-SPACE'
resource_group = 'ResourceGp-VK'
subscription_id = '758d9519-6a50-420c-a094-611f42144a79'
location = 'westus2'


try:
ws = Workspace.get(
        name=workspace_name,
subscription_id=subscription_id,
resource_group=resource_group,
        auth=sp,
    )

except:
    # this call might take a minute or two.
print("Creating new workspace")
ws = Workspace.create(
        name=workspace_name,
subscription_id=subscription_id,
resource_group=resource_group,
        # create_resource_group=True,
        location=location,
        auth=sp,
    )

# print Workspace details
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")
