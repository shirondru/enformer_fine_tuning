import numpy as np
import sys
import os
from scipy.stats import pearsonr, false_discovery_control
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
from .ism_performer import *

#import warning that will happen if model predicts same value for every person (when model is really bad at a gene)
#or if nobody (or everybody) has a particular snp during driver analysis
import warnings
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

def get_genotypes(gene,window,donors):
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,"../data")
    vcf_path = os.path.join(DATA_DIR,"GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz")
    vcf_reader = vcfpy.Reader.from_path(vcf_path,tabix_path = vcf_path + '.csi')
    enformer_regions = pd.read_csv(os.path.join(DATA_DIR,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    gene_info = enformer_regions[enformer_regions['gene_name'] == gene]    
    variant_dict_keys = ['region','chrom','pos0','pos1','ref','alt'] + donors
    
    if gene_info.shape[0] > 0:
        variant_dict = {k: [] for k in variant_dict_keys}
        region = get_window_around_TSS(window,gene_info) #define region +/- window of TSS, in order to get all observed snps in donors within +/- window of TSS
        for i,record in enumerate(vcf_reader.fetch(region)):
            af = record.INFO['AF'][0]
            if af > 0: #Everyone should have each of these SNPs at least once, but don't include any that are unobserved
                ref = record.REF
                alt = record.ALT[0].value
                chrom = record.CHROM
                variant_dict['region'].append(region)
                variant_dict['chrom'].append(chrom)
                variant_dict['pos1'].append(record.POS)
                variant_dict['pos0'].append(record.POS - 1)
                variant_dict['ref'].append(ref)
                variant_dict['alt'].append(alt)
                variant_dict['gene_name'] = gene
                [variant_dict[call.sample].append(call.data['GT']) for call in record.calls if call.sample in donors]
                    
        return pd.DataFrame(variant_dict)
    else:
        raise Exception(f"Gene {gene} not in enformer regions")

def plot_driver_selection(sum_attr_x_genotype, snp_attr_x_genotype,forward_sum_attr_x_genotype, model_preds,  peartot, snp_pearson,pearson_forward, model_outdir,i,filename):
    # fig,axes = plt.figure(figsize = (4,4))
    # ax = fig.add_subplot(111)
    fig, axes = plt.subplots(3, 1,sharex = True, sharey = True,figsize = (12,12))
    for ax in axes:
        ax.set_ylabel('Enformer prediction')
        ax.set_xlabel('Sum prediction')
    axes[0].scatter(sum_attr_x_genotype, model_preds['y_pred'],color = 'grey') #plot full sum of attributions x genotype among all SNPs vs full predictions
    axes[0].set_title(f"Linear Approximation Using Full Sum of All SNPs: {round(peartot,2)}")
    
    axes[1].scatter(snp_attr_x_genotype,model_preds['y_pred'], color = 'goldenrod') #plot genotype x ism for this SNP and the correlation that gives with full model prediction
    axes[1].set_title(f"Linear Approximation Using Only Current SNP: {round(snp_pearson,2)}")

    axes[2].scatter(forward_sum_attr_x_genotype,model_preds['y_pred'], color = 'navy')
    axes[2].set_title(f"Linear Approximation Using Current Sum of SNPs: {round(pearson_forward,2)}")
   
    parsed_filename = filename.split('.csv')[0]
    save_path = os.path.join(model_outdir,"DriverSelectionFigures",parsed_filename)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path,f"DriverSelection_{i}.png"),dpi = 200, bbox_inches = 'tight')
    plt.close()
def forward_selection(peartot,attr_x_genotype,plot_selection,model_preds,model_outdir,sum_attr_x_genotype,filename):
    """
    Adapted from https://github.com/mostafavilabuw/EnformerAssessment/blob/main/enformer_analysis/select_drivers.py
    Procedure:
    1) Iterate through all observed SNPs in these people and their ISM scores. The order of SNPs is sorted by ISM Attribution (largest magnitude comes first)
    Using the pre-calculated attr_x_genotype matrix, which is the element-wise product of each SNPs genotype (per person) and ISM score, take the sum of values for each SNP up to and including the currrent snp, in order of descending raw ISM Values (attr_x_genotype is sorted in this way)
    attr_x_genotype has SNPs as rows and donors as columns. Genotypes x ISM scores are the elements. 
    2) For each person, take the sum over the attr_x_genotype scores. This is a linear approximation of the attribution of these variants on the model's prediction
    3) Take the pearson correlation between these and the full models prediction. 
    4) Compare that value to the pearson correlation you get when you use the linear approximation of all observed SNPs. What proportion does it make up?
    5) For just the current SNP, get its linear approximation value and take correlation with full model predictions
    6) Check if (1) the current combination of SNPs increases correlation with Enformer's ful predictions by 5% more than the previous combination.
       and (2) if the current SNP on its own has a linear approximation that is correlated with the full predicitons and (3) if the pearson is significant after bonferroni correction
    7) If so, it is a driver
    """
    old_forward_contribution = 0
    drivers = []
    forward_selection_dict = {'forward_contribution':[],'old_forward_contribution':[],'pearson_forward':[],'snp_pearson':[],'snp_pearson_p':[],'variant':[]}
    
    for i, variant in enumerate(attr_x_genotype.index):
        
        #for all SNPs until this point, take sum of attributions times genotype to see if this correlates with full predictions in same people. 
        #This returns sum of attribution for all SNPs (until this point) for each person
        forward_sum_attr_x_genotype = np.sum(attr_x_genotype.iloc[:i+1, :],axis = 0).reindex(model_preds.index) #then reindex to make sure order of donors and their linear approx is same as in model_preds
        
        pearson_forward = pearsonr(forward_sum_attr_x_genotype,model_preds['y_pred'])[0]
        forward_contribution = pearson_forward / peartot #What % of the correlation formed by the linear approximation do the current SNPs add
    
        snp_attr_x_genotype = attr_x_genotype.iloc[i,:].reindex(model_preds.index) #ISM attribution times genotype for this SNP, for all people
        snp_pearson, snp_pearson_p = pearsonr(snp_attr_x_genotype,model_preds['y_pred'],alternative = 'greater')
    
        #1) Does this combination of SNPs inrease correlation with Enformers full predictions by 5% more than the previous  combination?
        #2) Is this SNP individually positively with Enformer's prediction
        if ((forward_contribution - old_forward_contribution) > 0.05) and (snp_pearson > 0) and (snp_pearson_p < 0.01/len(attr_x_genotype.index)):
            drivers.append(variant)
            if plot_selection == 'drivers':
                plot_driver_selection(sum_attr_x_genotype, snp_attr_x_genotype,forward_sum_attr_x_genotype, model_preds,  peartot, snp_pearson,pearson_forward, model_outdir,variant,filename)

        if plot_selection == 'all':
            plot_driver_selection(sum_attr_x_genotype, snp_attr_x_genotype,forward_sum_attr_x_genotype, model_preds,  peartot, snp_pearson,pearson_forward, model_outdir,i,filename)
        
        forward_selection_dict['forward_contribution'].append(forward_contribution)
        forward_selection_dict['old_forward_contribution'].append(old_forward_contribution)
        forward_selection_dict['snp_pearson'].append(snp_pearson)
        forward_selection_dict['snp_pearson_p'].append(snp_pearson_p)
        forward_selection_dict['variant'].append(variant)
        forward_selection_dict['pearson_forward'].append(pearson_forward)
        old_forward_contribution = np.nan_to_num(forward_contribution,0)
        
    df = pd.DataFrame(forward_selection_dict)
    df['is_driver'] = df['variant'].isin(drivers)
    df['LinearApproximationCorrelation'] = peartot
    return df
   
def forward_selection_with_only_drivers(peartot,attr_x_genotype,plot_selection,model_preds,model_outdir,sum_attr_x_genotype,filename):
    """
    Adapted from https://github.com/mostafavilabuw/EnformerAssessment/blob/main/enformer_analysis/select_drivers.py
    Same as `forward_selection` except instead of each SNP being iteratively added to the sum, they are only added to the sum if they are a driver
    
    The old forward contribution may go up and down until a driver is found. Then it shouldn't decrease past that baseline
    """
    old_forward_contribution = 0
    drivers = []
    forward_selection_dict = {'forward_contribution':[],'old_forward_contribution':[],'pearson_forward':[],'snp_pearson':[],'snp_pearson_p':[],'variant':[]}
    
    for i, variant in enumerate(attr_x_genotype.index):
    
        forward_sum_attr_x_genotype = np.sum(attr_x_genotype.loc[drivers + [variant], :],axis = 0).reindex(model_preds.index) #then reindex to make sure order of donors and their linear approx is same as in model_preds
        
        pearson_forward = pearsonr(forward_sum_attr_x_genotype,model_preds['y_pred'])[0]
        forward_contribution = pearson_forward / peartot #What % of the correlation formed by the linear approximation do the current SNPs add
    
        snp_attr_x_genotype = attr_x_genotype.iloc[i,:].reindex(model_preds.index) #ISM attribution times genotype for this SNP, for all people
        snp_pearson, snp_pearson_p = pearsonr(snp_attr_x_genotype,model_preds['y_pred'],alternative = 'greater')

        if plot_selection == 'all':
            plot_driver_selection(sum_attr_x_genotype, snp_attr_x_genotype,forward_sum_attr_x_genotype, model_preds,  peartot, snp_pearson,pearson_forward, model_outdir,i,filename)
        
        forward_selection_dict['forward_contribution'].append(forward_contribution)
        forward_selection_dict['old_forward_contribution'].append(old_forward_contribution)
        forward_selection_dict['snp_pearson'].append(snp_pearson)
        forward_selection_dict['snp_pearson_p'].append(snp_pearson_p)
        forward_selection_dict['variant'].append(variant)
        forward_selection_dict['pearson_forward'].append(pearson_forward)

        if ((forward_contribution - old_forward_contribution) > 0.05) and (snp_pearson > 0) and (snp_pearson_p < 0.01/len(attr_x_genotype.index)):
            drivers.append(variant)
            if plot_selection == 'drivers':
                plot_driver_selection(sum_attr_x_genotype, snp_attr_x_genotype,forward_sum_attr_x_genotype, model_preds,  peartot, snp_pearson,pearson_forward, model_outdir,variant,filename)

            old_forward_contribution = forward_contribution
    df = pd.DataFrame(forward_selection_dict)
    df['is_driver'] = df['variant'].isin(drivers)
    df['LinearApproximationCorrelation'] = peartot
    return df

def eval_driver_performance(model_outdir,driver_filename, attr_x_genotype,model_preds):
    driver_results = pd.read_csv(os.path.join(model_outdir,driver_filename))
    drivers = driver_results[driver_results['is_driver'] == True]['variant']

    forward_sum_attr_x_genotype = np.sum(attr_x_genotype.loc[drivers, :],axis = 0).reindex(model_preds.index) #then reindex to make sure order of donors and their linear approx is same as in model_preds
    assert all(model_preds.index == forward_sum_attr_x_genotype.index), "Donors must be in same order before taking correlation"
    pearson_driver_observed = pearsonr(forward_sum_attr_x_genotype,model_preds['y_true'])[0].item()
    r2_driver_observed = r2_score(model_preds['y_true'],forward_sum_attr_x_genotype)
    metrics_dict = {
        'pearson_driver_observed' : [pearson_driver_observed],
        'r2_driver_observed' : [r2_driver_observed]
     }
    pd.DataFrame(metrics_dict).to_csv(os.path.join(model_outdir,"DriverPerformance.csv"))
    model_preds.join(forward_sum_attr_x_genotype.rename('driver_sum')).to_csv(os.path.join(model_outdir,"DriverPredictions.csv"))
    
def select_and_evaluate_drivers(ism_results,desired_seq_len,gene_name,donors, model_preds, plot_selection,model_outdir, driver_method,filename,select_drivers,evaluate_drivers):
    gt_map = {
        '0/0': 0 ,
        '0/1': 1, 
        '1/1': 2, 
        './.': 0
        }

    window = desired_seq_len // 2
    variant_df = get_genotypes(gene_name,window,donors)
    for donor in donors: #this also tests that all donors have called genotypes for each of the SNPs, else an error will be raised here
        variant_df[donor] = variant_df[donor].map(gt_map) #convert genotype to int (e.g., 0/0 -> 0; 1/1 -> 2)


    before_shape = ism_results.shape[0]
    ism_results = ism_results.merge(variant_df, on = ['chrom','pos0','ref','alt','gene_name']) #merge genotype calls for each donor per SNP to the same dataframe as the ISM values for  each SNP
    after_shape = ism_results.shape[0]
    assert before_shape == after_shape, "Number of variants in your ISM results are now missing after merging genotypes in!"

    ism_results['variant'] = ism_results['chrom'] + '_' + ism_results['pos0'].astype(str) + '_' + ism_results['ref'] + '_' + ism_results['alt']
    ism_results = ism_results.set_index('variant').sort_values(by = 'attr',key = abs,ascending = False) #sort by absolute value of attribution. Largest at top


    #Multiply genotype of each SNP in each donor element-wise by the ISM attribution (Alt - Ref) for the same SNP
    attr_x_genotype = ism_results[donors].multiply(ism_results['attr'],axis = 0) #axis 0 specifies genotype for each donor is multiplied by ISM value from same row in attr column (i.e, the same SNP)

    if select_drivers == 'true':
        #take sum of resulting dataframe across rows. Starting with a column of ISM * genotype values for each person, this returns the sum of ISM * Genotypes among all SNPs for each person. One final value per person
        #this value is the sum  of the attributions times genotype. It is a linear approximation of the model predictions. Linear approximation means additivity is assumed -- considering each SNP in isolation in ISM
        sum_attr_x_genotype =  np.sum(attr_x_genotype, axis = 0).reindex(model_preds.index)
        peartot = pearsonr(sum_attr_x_genotype,model_preds['y_pred'])[0] #how close the linear approximation is to the full model predictions

        if driver_method == 'forward_selection':
            df = forward_selection(peartot,attr_x_genotype,plot_selection,model_preds,model_outdir,sum_attr_x_genotype,filename)
        else:
            df = forward_selection_with_only_drivers(peartot,attr_x_genotype,plot_selection,model_preds,model_outdir,sum_attr_x_genotype,filename)
        df.to_csv(os.path.join(model_outdir,filename))
    if evaluate_drivers == 'true':
        assert os.path.exists(os.path.join(model_outdir,filename)), "Driver selection must precede driver evaluate"
        eval_driver_performance(model_outdir,filename, attr_x_genotype,model_preds)


def get_filenames_to_skip(skip_finished_runs, model_outdir):
    """
    return a list of filenames corresponding to finished experiments from this model if skip_finished_runs == 'true'. this list is empty if 'false'. The filename that will otherwise be made is checked against this list. 
    If it is not in this list, it will be made. So if the list is empty they will always be made.
    """
    if skip_finished_runs == 'true':
        finished_runs = os.listdir(model_outdir)
    else:
        finished_runs = []
    return finished_runs


def main():
    parser = argparse.ArgumentParser(description="Select Drivers from ISM attributions")
    parser.add_argument("--driver_method",type=str)
    parser.add_argument("--plot_selection",type=str)
    parser.add_argument("--path_to_metadata",type=str)
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--select_drivers",type=str)
    parser.add_argument("--evaluate_drivers",type=str)
    parser.add_argument("--skip_finished_runs",type=str)
    parser.add_argument("--outdir",nargs = '?',type=str)
    parser.add_argument("--ism_dir",nargs = '?',type=str,help = 'path to ISM results for observed SNPs around genes these models were trained/tested on')
    parser.add_argument("--subset",type=int,nargs = '?',help = "Metadata enumerating models will be split into n_subsets and this subset of models will be evaluated by the job.")
    parser.add_argument("--n_subsets",type=int,nargs = '?',help = "Metadata enumerating models will be split into n_subsets.")



    args = parser.parse_args()
    metadata = pd.read_csv(args.path_to_metadata)
    metadata = metadata.rename(columns = {'ID':'run_id'})
    model_type = args.model_type
    assert model_type in ['SingleGene','MultiGene','OligoGene']
    driver_method = args.driver_method
    plot_selection = args.plot_selection.lower()
    select_drivers = args.select_drivers.lower()
    evaluate_drivers = args.evaluate_drivers.lower()
    skip_finished_runs = args.skip_finished_runs.lower()
    outdir = args.outdir
    ism_dir = args.ism_dir
    subset = args.subset
    n_subsets = args.n_subsets
    assert str(plot_selection) in ['false','all','drivers']
    assert driver_method in ['forward_selection','forward_selection_with_only_drivers']
    assert select_drivers in ['false','true']
    assert evaluate_drivers in ['false','true']
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    if not ism_dir: 
        ism_dir = os.path.join(cwd,f'../results/PerformerISM')
    if not outdir:
        outdir = os.path.join(cwd,f'../results/PerformerDriverSelection/{model_type}')
    

    

    if n_subsets:
        n_subsets = int(n_subsets) #ensure read as int
        subset = int(subset)
        metadata = np.array_split(metadata,int(n_subsets))[int(subset)]

    


    for idx, row in metadata.iterrows():
        run_id = row['run_id']
        tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',') #configure as a list of strings
        assert len(tissues_to_train) == 1, "This script is currently only supported for single-tissue models"
        tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')
        ism_result_dir = os.path.join(ism_dir,f"{tissue_str}Models")
        desired_seq_len = int(row['seq_length'])
        results_dir = row['save_dir']
        pred_results_file = [file for file in os.listdir(results_dir) if 'in_test_donors' in file and 'Prediction_Results' in file]
        assert len(pred_results_file) == 1, "There should only be 1 file containing prediction results in test donors"
        pred_results_file = pred_results_file[0]

        
        model_results_dir = os.listdir(os.path.join(ism_result_dir,model_type, run_id))
        for idx2, ism_result_file in enumerate(model_results_dir):
            if tissues_to_train == ['Brain - Cortex']: #select drivers using fully held out gtex cohort, assuming model was trained in ROSMAP GTEx
                name = f"{run_id}_AllGTEx"
            elif tissues_to_train == ['Whole Blood']:
                name = f"{run_id}_TestSet"
            else:
                raise Exception(f"Tissue {tissues_to_train} is not supported!")
            gene_name = ism_result_file.split('_')[0]
            model_outdir = os.path.join(outdir,driver_method,name)
            filename = f'{gene_name}_{name}_{driver_method}_ISMDrivers.csv'
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            finished_runs = get_filenames_to_skip(skip_finished_runs,model_outdir)
            if filename not in finished_runs:
                print(f"On ism result {idx2} of {len(model_results_dir)} of Model {idx}/ {metadata.shape[0]} Using Method {driver_method}")
                ism_results = pd.read_csv(os.path.join(ism_result_dir,model_type,run_id,ism_result_file))
                ism_results['attr'] = ism_results['alt_pred'] - ism_results['ref_pred']
                
                model_preds = pd.read_csv(os.path.join(results_dir, pred_results_file))
                model_preds = model_preds[model_preds['gene'] == gene_name]
                donors = list(model_preds['donor'].unique())
                model_preds = model_preds.set_index('donor') #set donor as index to allow for easy re-indexing of pd.Series objects later 
                try:
                    select_and_evaluate_drivers(ism_results,desired_seq_len,gene_name,donors, model_preds, plot_selection,model_outdir, driver_method,filename,select_drivers,evaluate_drivers)
                except Exception as e:
                    print(e)
            else:
                print(f"Skipping {filename}")
            sys.stdout.flush()

if __name__ == '__main__':
    main()