import os
from os.path import join, basename
import subprocess
import numpy as np
from glob import glob


class RunAllModesTrainPredict:

    def __init__(self, predict_fold=None, task_number_range=range(14, 20), cuda_device_num=0):
        os.environ['RESULTS_FOLDER'] = r'D:\users\Yuval\test\nnUNet\nnunet\nnUNet_trained_models'
        os.environ['nnUNet_raw_data_base'] = r'D:\users\Yuval\test\nnUNet\nnunet\nnUNet_raw_data_base'
        os.environ['nnUNet_preprocessed'] = r'D:\users\Yuval\test\nnUNet\nnunet\nnUNet_preprocessed'
        os.environ['nnUNet_n_proc_DA'] = '28'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device_num)
        self.train_script_path = r'D:\users\Yuval\test\nnUNet\nnunet\run\run_training.py'
        self.predict_script_path = r'D:\users\Yuval\test\nnUNet\nnunet\inference\predict.py'
        self.preprocess_script_path = r'D:\users\Yuval\test\nnUNet\nnunet\experiment_planning\nnUNet_plan_and_preprocess.py'
        self.data_folder = r'D:\users\Yuval\test\nnUNet\nnunet\nnUNet_raw_data_base\nnUNet_raw_data'
        self.models_dir = r'D:\users\Yuval\test\nnUNet\nnunet\nnUNet_trained_models\nnUNet\3d_fullres'
        self.model_name = r'nnUNetTrainerV2__nnUNetPlansv2.1'
        self.architecture_name = '3d_fullres'
        self.trainer_name = '3d_fullres nnUNetTrainerV2'
        self.task_number_range = task_number_range  # range(14, 16)
        self.fold_range = [0]
        self.predict_fold = predict_fold
        self.task_list = []

    def create_predict_run_command(self, task_name, input_data_folder_name, output_data_folder_name):
        input_dir = join(self.data_folder, task_name, input_data_folder_name)
        output_dir = join(self.data_folder, task_name, output_data_folder_name)
        model_dir = join(self.models_dir, task_name, self.model_name)
        if self.predict_fold in range(6):
            run_command_list = ['python', self.predict_script_path, '-i', input_dir, '-o', output_dir, '-m', model_dir,
                                '-f', str(self.predict_fold), '--save_npz']
        else:
            self.predict_fold = 'CV' if self.predict_fold is None else self.predict_fold
            run_command_list = ['python', self.predict_script_path, '-i', input_dir, '-o', output_dir, '-m', model_dir,
                                '--save_npz']
        return run_command_list

    def create_train_run_command(self, task_number, fold_number):
        run_command_list = ['python', self.train_script_path, self.architecture_name, self.trainer_name,
                            str(task_number), str(fold_number)]
        return run_command_list

    def create_preprocess_run_command(self, task_number):
        run_command_list = ['python', self.preprocess_script_path, '-t', str(task_number)]
        return run_command_list

    def get_task_name(self, task_num):
        task_name = basename(glob(join(self.data_folder, '**', f'Task*{task_num:03}*'), recursive=True)[0])
        return task_name

    def run_all_predictions(self, data_folder: str = 'imagesTs'):
        self.task_list = [self.get_task_name(task_num) for task_num in self.task_number_range]
        for task_name in self.task_list:
            run_command_list = self.create_predict_run_command(
                task_name=task_name, input_data_folder_name=data_folder,
                output_data_folder_name=join(data_folder, f'Results_fold_{self.predict_fold}'))
            process = subprocess.Popen(run_command_list)
            process.wait()
            print(f'Finished running prediction for task: {task_name}')

    def run_all_predictions_parallel(self, data_folder: str = 'imagesTs'):
        self.task_list = [self.get_task_name(task_num) for task_num in self.task_number_range]
        for task_name in self.task_list:
            run_command_list = self.create_predict_run_command(
                task_name=task_name, input_data_folder_name=data_folder,
                output_data_folder_name=join(data_folder, f'Results_fold_{self.predict_fold}'))
            process = subprocess.Popen(run_command_list)
            return process

    def run_all_training(self):
        self.task_list = [self.get_task_name(task_num) for task_num in self.task_number_range]
        for task_num, task_name in zip(self.task_number_range, self.task_list):
            for fold_num in self.fold_range:
                run_command_list = self.create_train_run_command(task_number=task_num, fold_number=fold_num)
                process = subprocess.Popen(run_command_list)
                process.wait()
                print(f'Finished running training for task: {task_name}')

    def run_all_preprocessing(self):
        self.task_list = [self.get_task_name(task_num) for task_num in self.task_number_range]
        for task_num, task_name in zip(self.task_number_range, self.task_list):
            run_command_list = self.create_preprocess_run_command(task_number=task_num)
            process = subprocess.Popen(run_command_list)
            process.wait()
            print(f'Finished running preprocessing for task: {task_name}')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


folds_bool = True
task_number_range_list = range(1, 2)  # range(8, 14)
cuda_device_num_list = range(3)
child_processes_list = list()
task_groups = list(chunks(task_number_range_list, 3))
if folds_bool:
    for fold in range(0, 5):
        for task_group in task_groups:
            for task_num_range, cuda_dev_num in zip(task_group, cuda_device_num_list):
                predict_cls = RunAllModesTrainPredict(predict_fold=fold,
                                                      task_number_range=range(task_num_range, task_num_range + 1),
                                                      cuda_device_num=cuda_dev_num)
                child_process = predict_cls.run_all_predictions_parallel(data_folder=join('imagesTr'))
                # child_process = predict_cls.run_all_predictions_parallel(data_folder=join('brats_processing_test'))
                child_processes_list.append(child_process)
                # predict_cls.run_all_preprocessing()
                # predict_cls.run_all_training()
            for child_process in child_processes_list:
                child_process.wait()
else:
    for task_group in task_groups:
        for task_num_range, cuda_dev_num in zip(task_group, cuda_device_num_list):
            predict_cls = RunAllModesTrainPredict(task_number_range=range(task_num_range, task_num_range + 1),
                                                  cuda_device_num=cuda_dev_num)
            child_process = predict_cls.run_all_predictions_parallel(data_folder=join('imagesTs'))
            # child_process = predict_cls.run_all_predictions_parallel(data_folder=join('brats_train_data', f'CV'))
            # child_process = predict_cls.run_all_predictions_parallel(data_folder=join('brats_processing_test'))
            child_processes_list.append(child_process)
            # predict_cls.run_all_preprocessing()
            # predict_cls.run_all_training()
        for child_process in child_processes_list:
            child_process.wait()



# Single case
# predict_cls = RunAllModesTrainPredict()
# predict_cls.run_all_predictions(data_folder=join('brats_2022_val'))
# # predict_cls.run_all_preprocessing()
# # predict_cls.run_all_training()
