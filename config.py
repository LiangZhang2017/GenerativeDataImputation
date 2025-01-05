from AmbientGAN.AmbientGAN_model import AmbientGAN
from EmbeddingGAIN.Embedding_GAIN_model import Embedding_GAIN
from GAIN.GAIN_model import GAIN
from helper import read_as_tensor, calculate_sparsity,sparsity_explore,save_tensor

from sparsity_adjust import Sparsity_Adjust

import tensorflow as tf
import os
import pandas as pd

from TC.Standard_TC_model import Standard_TC
from TC.CP_model import CPDecomposition
from TC.BPTF_model import BPTF

from VAE.VAE_model import VAE
from VAE.New_VAE_model import New_VAE
from GAN.GAN_model import GAN
from GAN.New_GAN_model import New_GAN

from CGAN.CGAN_model import CGAN
from AE.AE_model import AE

from InfoGAN.InfoGAN_model import InfoGAN
from BetaVAE.BetaVAE_model import BetaVAE
from FactorVAE.FactorVAE_model import FactorVAE

from AE.AE_helper import save_results

'''
'learning_stage':
CSAL: 'Medium', 'Easy', 'Hard'
MATHia: "scale_drawings_3","Geometry_MATHia","analyzing_models_2step_integers","worksheet_grapher_a1_patterns_2step_expr"
'''

class Model_Config:
    def __init__(self,args):
        self.args=args

    def model_factory(self,model_name, tensor, set_parameters):
        model_constructors = {
            "Standard_TC": lambda: Standard_TC(tensor, set_parameters),
            "Standard_CPD": lambda: CPDecomposition(tensor, set_parameters),
            "BPTF": lambda: BPTF(tensor, set_parameters),
            "AE": lambda: AE(tensor, set_parameters),
            "VAE": lambda: New_VAE(tensor, set_parameters),  # Assuming New_VAE is the desired class
            "GAN": lambda: New_GAN(tensor, set_parameters),  # Assuming New_GAN is the desired class
            "CGAN": lambda: CGAN(tensor, set_parameters),
            "GAIN": lambda: GAIN(tensor, set_parameters),
            "AmbientGAN": lambda: AmbientGAN(tensor, set_parameters),
            "Embedding_GAIN": lambda: Embedding_GAIN(tensor, set_parameters),
            "InfoGAN":lambda: InfoGAN(tensor, set_parameters),
            "BetaVAE": lambda: BetaVAE(tensor, set_parameters),
            "FactorVAE":lambda: FactorVAE(tensor, set_parameters),
        }
        constructor = model_constructors.get(model_name)
        return constructor() if constructor else None

    def generate_paradic(self):

        if self.args.Imput_model[0]=='Standard_TC':
            print("Standard setting")

            course=self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            features_dim=6

            lambda_t=0.00001 # lambda_t and lambda_q are hyper-parameters to control the weights of regularization term of T and S.
            lambda_q=0.001
            lambda_bias=0.001  # can be set as False
            lambda_w=0.1  #for rank-based TF
            lr=0.0001
            max_iter=100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'lambda_t': lambda_t,
                    'lambda_q': lambda_q,
                    'lambda_bias': lambda_bias,
                    'lambda_w': lambda_w,
                    'lr': lr,
                    'features_dim':features_dim,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='Standard_CPD':
            print("Standard setting")

            course=self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            features_dim=6

            lambda_t=0.00001 # lambda_t and lambda_q are hyper-parameters to control the weights of regularization term of T and S.
            lambda_q=0.001
            lambda_bias=0.001   # can be set as False
            lambda_w=0.1  #for rank-based TF
            lr=0.0001
            max_iter=100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'lambda_t': lambda_t,
                    'lambda_q': lambda_q,
                    'lambda_bias': lambda_bias,
                    'lambda_w': lambda_w,
                    'lr': lr,
                    'features_dim':features_dim,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='BPTF':
            print("Parameters for BPTF")

            course=self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            features_dim=6

            lambda_u = 0.001
            lambda_v = 0.001
            lambda_x = 0.001
            lambda_bias = 0.001  # can be set as False
            max_iter=100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'lambda_u': lambda_u,
                    'lambda_v': lambda_v,
                    'lambda_x': lambda_x,
                    'lambda_bias':lambda_bias,
                    'features_dim':features_dim,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='AE':
            print("AE setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc","f1_core"]

            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'mode': 'one-shot', # one shot or few shot
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='VAE':
            print("VAE setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='GAN':
            print("GAN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.0001
            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr':lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='CGAN':
            print("CGAN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.00001
            max_iter = 2

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr':lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='GAIN':
            print("GAIN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.00001
            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr':lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='Embedding_GAIN':
            print("Embedding_GAIN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.0001
            max_iter = 1000

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr':lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='AmbientGAN':
            print("AmbientGAN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.0001
            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr':lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning':'attempt_wise'
                    }

        if self.args.Imput_model[0]=='InfoGAN':
            print("InfoGAN setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            lr = 0.0001
            max_iter =100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'lr': lr,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning': 'attempt_wise'
                    }

        if self.args.Imput_model[0]=='BetaVAE':
            print("BetaVAE setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning': 'attempt_wise'
                    }

        if self.args.Imput_model[0]=='FactorVAE':
            print("Factor setting")

            course = self.args.Course[0]
            lesson_id = self.args.Lesson_Id[0]
            model_str = self.args.Imput_model[0]
            validation = True  # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]

            max_iter = 100

            para = {'lesson_id': lesson_id,
                    'course': course,
                    'model_str': model_str,
                    'max_iter': max_iter,
                    'metrics': metrics,
                    'validation': validation,
                    'learning_stage': 'Medium',
                    'is_rank': True,
                    'KCmodel': 'Unique',  # 'Unique' or 'Single'
                    'Use_KC': False,
                    'sparsity_adjust': True,
                    'SparsityPruning': 'attempt_wise'
                    }

        return para

    def main(self):
        print("start run model")

        set_parameters=self.generate_paradic()

        #read the dataset as in tensor framework
        raw_T, numpy_T=read_as_tensor(self.args,set_parameters)

        tf_tensor=tf.convert_to_tensor(raw_T)  # students*questions*attempts,obs
        print("tf_tensor.shape is {}".format(tf_tensor.shape))

        # Save the original sparse tensor
        save_tensor(tf_tensor,set_parameters)

        model_name=set_parameters['model_str']
        print("model_name is {}".format(model_name))

        collected_results=[]

        sparsity_adjust=set_parameters['sparsity_adjust']
        mode=set_parameters['SparsityPruning']

        if sparsity_adjust is True:
            prune_sparsity_dict=sparsity_explore(tf_tensor,numpy_T,mode)

            for i, data in prune_sparsity_dict.items():
                prune_tensor = data['prune_tensor']
                prune_numpy_T = data["prune_numpy_T"]
                sparsity = data['sparsity']
                prune_slice_number = data['prune_slice_number']
                set_parameters['prune_slice_number']=prune_slice_number

                model = self.model_factory(model_name, prune_tensor, set_parameters)
                if model:
                    print(f"Start of {model_name} model")

                    for model_iter in range(5):
                        eval_results, fold_summary = model.RunModel(model_iter)  # Assuming a consistent method name like RunModel()
                        # Append the results with the current sparsity level
                        for fold, mae, rmse, rse, auc, cross_entropy,iter, version in eval_results:
                            collected_results.append({
                                'Version': version,
                                'Model_iter': model_iter+1,
                                'Prune_slice_number': prune_slice_number,
                                'Sparsity': sparsity,
                                'Elapsed_time': fold_summary.get(fold, "Not found"),
                                'Fold': fold,
                                'Itr_num': iter,
                                'MAE': mae.numpy() if isinstance(mae, tf.Tensor) else mae,
                                'RMSE': rmse.numpy() if isinstance(rmse, tf.Tensor) else rmse,
                                'RSE': rse.numpy() if isinstance(rse, tf.Tensor) else rse,
                                'AUC': auc.numpy() if isinstance(auc, tf.Tensor) else auc,
                                'CrossEntropy': cross_entropy
                            })
                else:
                    print(f"Model name {model_name} not recognized.")

            # Save the results
            save_results(collected_results, model_name, set_parameters)