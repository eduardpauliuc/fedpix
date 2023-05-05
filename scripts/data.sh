#clients=(europe asia us)

az configure --defaults group="dppix"

# Download the Pneumonia dataset
#pip install kaggle
#pip install split-folders
#kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /tmp
#unzip -q /tmp/chest-xray-pneumonia.zip -d /tmp

# split in train, val and test
#splitfolders --output /tmp/chest_xray_tvt/ --ratio .8 .1 .1 --seed 33 --move -- /tmp/chest_xray/train

# store central dataset to asia hospital

#az ml data create --workspace-name Asia-Site \
#                  --name maps-central \
#                  --path ${ROOT_DIR} \
#                  --type uri_folder
#
#stages=( train test val )
#classes=( PNEUMONIA NORMAL)
#
## Create folders
#for client in "${clients[@]}"; do
#    mkdir /tmp/chest_xray_$client
#    for stage in "${stages[@]}"; do
#        mkdir /tmp/chest_xray_$client/$stage
#        for class in "${classes[@]}"; do
#            mkdir /tmp/chest_xray_$client/$stage/$class
#        done
#    done
#done

## Copy data to client folders
#i=0
#for file in $(find /tmp/chest_xray_tvt -name '*.jpeg'); do
#    classnr=$(( i % 3 ))
#    cp $file ${file/chest_xray_tvt/chest_xray_${clients[classnr]}}
#    i=$((i+1))
#done


ROOT_DIR="/Users/eduardpauliuc/PycharmProjects/federated"
ASIA_DIR="${ROOT_DIR}/maps_az/site-1"
EUROPE_DIR="${ROOT_DIR}/maps_az/site-2"
US_DIR="${ROOT_DIR}/maps_az/site-3"

#Upload data assets
az ml data create --workspace-name Asia-Site \
                  --name pneumonia-dataset \
                  --path ${ASIA_DIR} \
                  --type uri_folder

az ml data create --workspace-name Europe-Site \
                  --name pneumonia-dataset \
                  --path ${EUROPE_DIR} \
                  --type uri_folder

az ml data create --workspace-name US-Site \
                  --name pneumonia-dataset \
                  --path ${US_DIR} \
                  --type uri_folder