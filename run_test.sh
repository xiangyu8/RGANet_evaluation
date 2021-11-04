DATA_PATH="data/suction_grasping"
IMG_PATH="$DATA_PATH/color-input"
LABEL_PATH="$DATA_PATH/label"
CHECKPOINT_PATH="checkpoint"

### --------model set 1---------
### model-backbone choices="fcn-resnet50 fcn-resnet101 deeplab-resnet50 deeplab-resnet101 deeplab-mobinenetv3 deeplab-mobilenetv2 stdc1-STDCNet813"
### for model set 1, select model-backbone from choices listed above.i
model="stdc1"
backbone="STDCNet813"
python seg_models.py -test -i $IMG_PATH -l $LABEL_PATH -c $CHECKPOINT_PATH/$model-$backbone/*.pt* -m $model -b $backbone


### --------model set 2---------
model_lib2="ccnet rganet hardnet shelfnet"
for model in $model_lib2
do
	python seg_models.py -test -i $IMG_PATH -l $LABEL_PATH -c $CHECKPOINT_PATH/$model/*.pt* -m $model -b $model
done

### --------model set 3---------
model_lib3="hrnet ddrnet"
for model in $model_lib3
do
	python seg_models.py -test -i $IMG_PATH -l $LABEL_PATH -c $CHECKPOINT_PATH/$model/*.pt* -m $model -b $model --cfg ./config/yamls/*$model*
done

