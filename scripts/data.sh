#clients=(europe asia us)

az configure --defaults group="dppix"

ROOT_DIR="/Users/eduardpauliuc/PycharmProjects/federated"
EU_N_DIR="${ROOT_DIR}/maps_az/site-1"
EU_W_DIR="${ROOT_DIR}/maps_az/site-2"
US_DIR="${ROOT_DIR}/maps_az/site-3"
#
#Upload data assets
az ml data create --workspace-name EUN-Site \
                  --name maps-dataset \
                  --path ${EU_N_DIR} \
                  --type uri_folder

az ml data create --workspace-name EUW-Site \
                  --name maps-dataset \
                  --path ${EU_W_DIR} \
                  --type uri_folder

az ml data create --workspace-name US-Site \
                  --name maps-dataset \
                  --path ${US_DIR} \
                  --type uri_folder