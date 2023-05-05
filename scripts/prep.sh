az extension add --name ml

az group create -n dppix -l northeurope

az vm create -n fedserver --image microsoft-dsvm:ubuntu-1804:1804-gen2:latest --authentication-type password --public-ip-address-dns-name dppix \
  --admin-password {PASSWORD}

RANDOMQUALIFIER=$RANDOM

az ml workspace create -n Asia-Site --location eastasia
az ml compute create --type ComputeInstance --name FedClientAsia$RANDOMQUALIFIER --workspace-name Asia-Site

az ml workspace create -n Europe-Site --location northeurope
az ml compute create --type ComputeInstance --name FedClientEurope$RANDOMQUALIFIER --workspace-name Eruope-Site

az ml workspace create -n US-Site --location eastus
az ml compute create --type ComputeInstance --name FedClientUS$RANDOMQUALIFIER --workspace-name US-Site




#{
#  "fqdns": "dppix.northeurope.cloudapp.azure.com",
#  "id": "/subscriptions/eba3ee9f-9743-43ea-a1b9-31c6c15580bc/resourceGroups/dppix/providers/Microsoft.Compute/virtualMachines/fedserver",
#  "location": "northeurope",
#  "macAddress": "00-0D-3A-B2-38-82",
#  "powerState": "VM running",
#  "privateIpAddress": "10.0.0.4",
#  "publicIpAddress": "40.85.89.20",
#  "resourceGroup": "dppix",
#  "zones": ""
#}
