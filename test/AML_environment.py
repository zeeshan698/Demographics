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

sp = ServicePrincipalAuthentication(tenant_id="b2ef2cdf-362a-48ec-8c76-321b322ed859", 
                                    service_principal_id="88226988-d739-4044-afb1-62220d8f507d", 
                                    service_principal_password="s2MeaVDdTqlikvQlf3X0.WP7-T3WIZEDmF") 

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
