
import argparse
from config import Model_Config

class DataImput:
    def __init__(self):

        '''
        Use the dataset in CSAL and MATHia for data imputation
        1. datapath: /dataset/CSAL/.. or /dataset/MATHia/..
        2. Lesson_Id
           "CSAL": "lesson17","lesson21","lesson20","lesson28" (too small dataset)
           "MATHia": 'worksheet_grapher_a1_patterns_2step_expr','scale_drawings_3','analyzing_models_2step_integers'
           "ASSISMENTS": 'assismentsmath2008-2009', '2012-2013-data-with-predictions-4-final' (filter version), 'assismentsmath2009-2010' (too small),
        3. Imput_model: "Standard_TC", "Standard_CPD", "BPTF", "GAN", "InfoGAN", "AmbientGAN", "GAIN", "Embedding_GAIN"
                        "AE", "VAE", "BetaVAE", "FactorVAE", "Adversarial Autoencoder" (AAE)
        4. Sparsity Levels: attempt-wise

        #Modification
        1. Original tensor
        2. Imputed tensor
        3. Tracking iteration
        4. Split a dataset into three subsets: training, testing, and validation
        5. Synthetic data

        Note:
            1. First Record all results for every iteration
            2. Original sparse tensor and dense tensor
        '''

        # github pad: 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

        parser=argparse.ArgumentParser(description='Arguments for Parameters Setting')
        parser.add_argument("--Course",nargs=1,type=str,default=['CSAL'])
        parser.add_argument("--data_path",nargs=2,type=str,default=['/dataset','/CSAL'])
        parser.add_argument("--Lesson_Id",nargs=1,type=str,default=['lesson21'])
        parser.add_argument("--Imput_model",nargs=1,type=str,default=['GAIN'])

        args=parser.parse_args()
        self.args=args


    def main(self):
        config=Model_Config(self.args)
        config.main()


if __name__ == '__main__':
    print("main")
    obj=DataImput()
    obj.main()