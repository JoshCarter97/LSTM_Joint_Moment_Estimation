
import torch
import torch.nn as nn
import numpy as np
import Training_Utility_Functions as Training_Utils
import os
import Dataset_Generator
import ML_Models
import random

# set the seed for reproducibility
def set_seed(seed):
    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for numpy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed)

    # If using CUDA, set the seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU.

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 1303
set_seed(seed)

base_path = "/app"  # this is the base path for this repo's location (for example within the docker container)


# read in the participant details from the npy file
participant_details = np.load(os.path.join(base_path, "Dataset", "Info_Files", "participant_characteristics.npy"), allow_pickle=True).item()


participants = ["P002", "P003", "P004", "P006", "P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015",
                "P016", "P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028",
                "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041",
                "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P049", "P050", "P051", "P052"]


def Run_LOSO(participants, input_variables, target_variable, side, save_folder_name):

    # loop through and create a list of participants in train and val set
    for participant in participants:

        trial_dict_path = os.path.join(base_path, "Dataset", "Info_Files", "processed_trials.npy")
        trial_dict = np.load(trial_dict_path, allow_pickle=True).item()

        EPOCHS = 6
        BATCH_SIZE = 8
        lr = 0.0005
        model = ML_Models.LSTM_Dropout(len(input_variables), hidden_size=512, input_dropout=0.2, linear_dropout=0.2,
                                          linear2_in=64, linear3_in=32, activation_function='relu')

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = nn.MSELoss(reduction='none')

        debug = False

        ###############################

        print(f"\n ============ Validating on participant {participant} ============")

        # current participant will be the validation participant (left out of training)
        val_participant = [participant]

        # The other 49 will be in the training set
        train_participants = participants.copy()
        train_participants.remove(participant)

        # just to speed up a full run through to make sure everything is working
        if debug:
            train_participants = train_participants[0:2]
            EPOCHS = 1

        # collect the training data
        train_generator = Dataset_Generator.DatasetGenerator(participants=train_participants,
                                                             input_variables=input_variables,
                                                             output_variables=target_variable,
                                                             side=side,
                                                             base_path=base_path)

        _, _, _ = train_generator.collect_dataset()
        train_input_data = train_generator.all_input_data
        train_output_data = train_generator.all_output_data
        train_contact_IDs = train_generator.all_contact_IDs

        # collect the validation data
        val_generator = Dataset_Generator.DatasetGenerator(participants=val_participant,
                                                           input_variables=input_variables,
                                                           output_variables=target_variable,
                                                           side=side,
                                                           base_path=base_path)

        val_input_data, val_output_data, val_contact_IDs = val_generator.collect_dataset()

        # scale the datasets to Zscores
        dataset_processor = Dataset_Generator.DatasetProcessor(input_variables, train_input_data, val_input_data)
        scaled_training_input_data, scaled_val_input_data, _, stats_list = dataset_processor.convert_to_zscores(group_FSR=False)

        # convert the data to pytorch datasets
        train_dataset = Training_Utils.Full_Load_Dataset(scaled_training_input_data, train_output_data, train_contact_IDs)
        val_dataset = Training_Utils.Full_Load_Dataset(scaled_val_input_data, val_output_data, val_contact_IDs)

        training_output_dict = Training_Utils.Run_Train_Val(train_dataset, val_dataset, model, optimiser, loss_func,
                                                            num_epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                            variable_names=input_variables, calculate_feature_importance=False,
                                                            num_permutations=100, units="Nm")

        # add the input variables, output variable, and scaling stats to the dictionary
        training_output_dict["input_variables"] = input_variables
        training_output_dict["output_variable"] = target_variable
        training_output_dict["stats_list"] = stats_list
        training_output_dict["side"] = side
        training_output_dict["val_contact_IDs"] = val_contact_IDs  # to be used to evaulate the results in more detail (order should be correct as no shuffle)

        # save the outputs of the training
        save_directory = os.path.join(base_path, save_folder_name)
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        save_path = os.path.join(save_directory, f"LOSO_{participant}inVal_training_outputs.npy")
        np.save(save_path, training_output_dict)
        print(f"Saved the training outputs to {save_path}")



### VARIABLES TO ADJUST ###

# repeating analysis completed in the paper
# all sensors
input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5",
         "FSR_6", "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15",
         "FSR_16", "Nurvv_CoPx", "Nurvv_CoPy", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z",
         "estimated_vGRF", "estimated_apGRF", "Sacrum_ACC_X", "Sacrum_ACC_Y", "Sacrum_ACC_Z", "Sacrum_GYRO_X",
         "Sacrum_GYRO_Y", "Sacrum_GYRO_Z", "T10_ACC_X", "T10_ACC_Y", "T10_ACC_Z", "T10_GYRO_X", "T10_GYRO_Y",
         "T10_GYRO_Z", "Wrist_ACC_X", "Wrist_ACC_Y", "Wrist_ACC_Z", "Wrist_GYRO_X", "Wrist_GYRO_Y",
         "Wrist_GYRO_Z", "Tibia_ACC_X", "Tibia_ACC_Y", "Tibia_ACC_Z", "Tibia_GYRO_X", "Tibia_GYRO_Y",
         "Tibia_GYRO_Z", "Thigh_ACC_X", "Thigh_ACC_Y", "Thigh_ACC_Z", "Thigh_GYRO_X", "Thigh_GYRO_Y",
         "Thigh_GYRO_Z", "Upperarm_ACC_X", "Upperarm_ACC_Y", "Upperarm_ACC_Z", "Upperarm_GYRO_X",
         "Upperarm_GYRO_Y", "Upperarm_GYRO_Z"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2All_Sensors_Ankle_Moment_HO")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2All_Sensors_Knee_Moment_HO")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2All_Sensors_Hip_Moment_HO")

# Lower body only
input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
         "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
         "Nurvv_CoPy", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "estimated_vGRF", "estimated_apGRF",
         "Sacrum_ACC_X", "Sacrum_ACC_Y", "Sacrum_ACC_Z", "Sacrum_GYRO_X", "Sacrum_GYRO_Y", "Sacrum_GYRO_Z",
         "Tibia_ACC_X", "Tibia_ACC_Y", "Tibia_ACC_Z", "Tibia_GYRO_X", "Tibia_GYRO_Y", "Tibia_GYRO_Z",
         "Thigh_ACC_X", "Thigh_ACC_Y", "Thigh_ACC_Z", "Thigh_GYRO_X", "Thigh_GYRO_Y", "Thigh_GYRO_Z"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2Lower_Body_Only_Ankle_Moment")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2Lower_Body_Only_Knee_Moment")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2Lower_Body_Only_Hip_Moment")

# T10 Sensor Model
input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
         "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
         "Nurvv_CoPy", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "estimated_vGRF", "estimated_apGRF",
         "T10_ACC_X", "T10_ACC_Y", "T10_ACC_Z", "T10_GYRO_X", "T10_GYRO_Y", "T10_GYRO_Z"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2Submission_T10_Ankle_Moment")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2Submission_T10_Knee_Moment")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2Submission_T10_Hip_Moment")

# Wrist Sensor Model
input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
         "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
         "Nurvv_CoPy", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "estimated_vGRF", "estimated_apGRF",
         "Wrist_ACC_X", "Wrist_ACC_Y", "Wrist_ACC_Z", "Wrist_GYRO_X", "Wrist_GYRO_Y", "Wrist_GYRO_Z"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2Submission_Wrist_Ankle_Moment")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2Submission_Wrist_Knee_Moment")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2Submission_Wrist_Hip_Moment")

# Foot Only
input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
         "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
         "Nurvv_CoPy", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z", "estimated_vGRF", "estimated_apGRF"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2Foot_Only_Ankle_Moment")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2Foot_Only_Knee_Moment")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2Foot_Only_Hip_Moment")


# # Wrist Only
# input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
#          "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
#          "Nurvv_CoPy", "estimated_vGRF", "estimated_apGRF",
#          "Wrist_ACC_X", "Wrist_ACC_Y", "Wrist_ACC_Z", "Wrist_GYRO_X", "Wrist_GYRO_Y", "Wrist_GYRO_Z"]
#
# Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Wrist_Only_Ankle_Moment")
# Run_LOSO(participants, input, ["Knee_Moment"], "left", "Wrist_Only_Knee_Moment")
# Run_LOSO(participants, input, ["Hip_Moment"], "left", "Wrist_Only_Hip_Moment")
#
#
# # No Sensors
# input = ["GCT", "mass", "speed", "incline", "insole_length", "FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6",
#          "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx",
#          "Nurvv_CoPy", "estimated_vGRF", "estimated_apGRF"]
#
# Run_LOSO(participants, input, ["Ankle_Moment"], "left", "No_Sensors_Ankle_Moment")
# Run_LOSO(participants, input, ["Knee_Moment"], "left", "No_Sensors_Knee_Moment")
# Run_LOSO(participants, input, ["Hip_Moment"], "left", "No_Sensors_Hip_Moment")

# scalar only
input = ["GCT", "mass", "speed", "incline", "insole_length"]

Run_LOSO(participants, input, ["Ankle_Moment"], "left", "Seed2Only_Discrete_Ankle_Moment")
Run_LOSO(participants, input, ["Knee_Moment"], "left", "Seed2Only_Discrete_Knee_Moment")
Run_LOSO(participants, input, ["Hip_Moment"], "left", "Seed2Only_Discrete_Hip_Moment")

