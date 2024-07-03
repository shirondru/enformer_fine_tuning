import numpy as np
import sys, os
from scipy.stats import pearsonr, false_discovery_control
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from ism_performer import *
import argparse
from select_drivers_performer import *

def get_enformer_eval(tissue,donor_fold,desired_seq_len,n_center_bins):
    cwd = os.getcwd()
    eval_dir = os.path.join(cwd,'../results/EnformerResults') #dir contaning Enformer evaluations on people from the test set
    if tissue == 'CAGE:blood, adult, pool1':
        results_dir = os.path.join(eval_dir,'FinalPaperWholeBlood')
        prediction_results_path = os.path.join(results_dir, f"Enformer_testGTExPredictions_DonorFold_{donor_fold}_WholeBloodTrainTestGenes_{desired_seq_len}bp_{n_center_bins}CenterBins.csv")
    elif tissue == 'CAGE:brain, adult': #use the GTEx predictions from Enformer evaluation. This will include all GTEx donors no matter the fold because this is a fully held out dataset for ROSMAP training
        results_dir = os.path.join(eval_dir,'FinalPaperBrainCortex')
        prediction_results_path = os.path.join(results_dir, f"Enformer_testGTExPredictions_DonorFold_{donor_fold}_BrainCortexTrainTestGenes_{desired_seq_len}bp_{n_center_bins}CenterBins.csv")
    else:
        raise Exception(f"Tissue {tissue} not supported!")
    return prediction_results_path
def main():
    """
    Takes Enformer predictions among people used to test the fine-tuned Enformer models, as well as Enformer's ISM values for observed SNPs in these people and finds SNPs that linearly approximate the full model's predictions (drivers)
    For brain, the held out people from GTEx were fully held out from fine-tuning. So the drivers will be the same for all 3 folds
    """
    parser = argparse.ArgumentParser(description="Select Drivers from ISM attributions")
    parser.add_argument("--driver_method",type=str)
    args = parser.parse_args()
    driver_method = args.driver_method

    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')

    plot_selection = 'drivers'
    outdir = os.path.join(cwd,'../results/EnformerDriverSelection')
    assert str(plot_selection).lower() in ['false','all','drivers']
    assert driver_method in ['forward_selection','forward_selection_with_only_drivers']
    tissues_to_train = ["CAGE:blood, adult, pool1","CAGE:brain, adult"]
    gene_paths_per_tissue = {
        'CAGE:brain, adult':os.path.join(data_dir,'genes/Brain_Cortex/all_brain_cortex_train_and_test_genes.txt'),
        'CAGE:blood, adult, pool1':os.path.join(data_dir,'genes/Whole_Blood/all_whole_blood_train_and_test_genes.txt')
    }

    tissue_name_map = {
        "CAGE:brain, adult": 'BrainCortex',
        'CAGE:blood, adult, pool1':'WholeBlood'
    }

    n_donor_folds = 3
    ism_result_dir = os.path.join(cwd,'../results/EnformerISM')

    finished_runs = os.listdir(os.path.join(outdir,driver_method)) #

    #for each tissue, ISM scores are in the file (ISM scores for each tissue on different rows)
    #So ISM scores for a gene could exist for both tissues but Enformer will only have been evaluated for one tissue. 
    #This loops through the ISM attribution files and grabs Enformer evaluations in people from GTEx with the same sequence length, num center bins, and donor fold
    #note that if the tissue is brain, donor fold is irrelevant because this is a fully held out dataset and CV was done over ROSMAP, so all donors will be the same and thus drivers will be the same because ISM attributions will always be the same
    
    
    for tissue in tissues_to_train:
        clean_tissue_name = tissue_name_map[tissue]
        genes_for_tissue = parse_gene_files(gene_paths_per_tissue[tissue]) #list of all train and test tissues used to fine-tune enformer models in the comparable tissue
        ism_result_files = os.listdir(ism_result_dir)
        for idx, ism_result_file in enumerate(ism_result_files):
            gene_name = ism_result_file.split('_')[0]
            if gene_name in genes_for_tissue: #ISM was saved for both tissues in the same files, even if the gene was only used for evaluation in one tissue. Skip genes that are not needed for a given tissue
                print(f"On ism result {idx} of {len(ism_result_files)} in {tissue}")
                desired_seq_len = int(ism_result_file.split('_')[4].strip('bp'))
                n_center_bins = int(ism_result_file.split('_')[-1].strip('CenterBins.csv'))
                ism_results = pd.read_csv(os.path.join(ism_result_dir,ism_result_file))
                ism_results = ism_results[ism_results['enformer_tissue'] == tissue] #keep only the desired tissue
                ism_results['attr'] = ism_results['alt_pred'] - ism_results['ref_pred']
                assert ism_results.shape[0] > 0
                assert ism_results['gene_name'].unique().item() == gene_name
                for donor_fold in range(n_donor_folds):
                    name = f"Enformer_{clean_tissue_name}_{desired_seq_len}bp_{n_center_bins}CenterBins_DonorFold{donor_fold}"
                    prediction_results_path = get_enformer_eval(tissue,donor_fold,desired_seq_len,n_center_bins)
                    if f"{gene_name}_{name}" not in finished_runs and os.path.exists(prediction_results_path): #perform analysis if its not already completed
                        model_preds = pd.read_csv(prediction_results_path)
                        model_preds = model_preds[model_preds['gene_name'] == gene_name]
                        donors = list(model_preds['donors'].unique())
                        model_preds = model_preds.set_index('donors') #set donor as index to allow for easy re-indexing of pd.Series objects later 
                        assert model_preds.shape[0] > 0
                        assert model_preds['enformer_tissue'].unique().item() == tissue
                        model_preds = model_preds.rename(columns = {'model_pred':'y_pred'})
                        select_drivers(ism_results,desired_seq_len,gene_name,donors, model_preds, plot_selection,outdir, driver_method,name)

if __name__ == '__main__':
    main()