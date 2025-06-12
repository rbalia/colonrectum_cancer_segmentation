import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd


print("Tensorflow and CUDA libraries/devices:")
print(f"\tTensorflow Version:  {tf.__version__}")
print(f"\tAvailable Devices:   {tf.config.list_physical_devices(device_type=None)}")
sys_details = tf.sysconfig.get_build_info()
print(f"\tCUDA Version:        {sys_details['cuda_version']}")
print(f"\tCUDNN Version:       {sys_details['cudnn_version']}")
print(f"\tCompute Capabily:    {sys_details['cuda_compute_capabilities']}")
print("\n")

print(f"Running Script '{os.path.basename(__file__)}' | Script Setting:")
if len(sys.argv) > 1 and sys.argv[1] in ["train", "test"]:
    script_mode = sys.argv[1]
    print(f"\tScript Mode: {script_mode}")
else:
    print("\tNot valid argument for 'script_mode'")
    script_mode = "test"
    print(f"\tSetting Default Mode: {script_mode} ")
    # exit(0)

if len(sys.argv) > 2 and sys.argv[2] in ["0", "1"]:
    target_gpu = sys.argv[2]
    print(f"\tSelected GPU: {target_gpu}")
else:
    print("\tNot valid argument for 'target_gpu'")
    target_gpu = "0"
    print(f"\tSetting Default GPU: {target_gpu}")

if len(sys.argv) > 3:
    experiment_name = sys.argv[3]
    print(f"\tExperiment name: {experiment_name}")
else:
    print("\tNot valid argument for 'experiment_name'")
    experiment_name = "default"
    print(f"\tSetting Default experiment name: {experiment_name}")
print("\n")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[int(target_gpu)], 'GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("Momory Growth setting failed")
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# logical_devices = tf.config.list_logical_devices('GPU')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 3000)

K.set_image_data_format('channels_last')


def get_param():
    return script_mode, target_gpu, experiment_name