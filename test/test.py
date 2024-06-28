import os
import sys
import numpy as np
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
from performer.train_gtex import *
from performer.eval_enformer_gtex import slice_enformer_pred
torch.use_deterministic_algorithms(True)
from performer.ism_performer import load_model
def test_train_dataloader():
    config_path = "../code/configs/blood_config.yaml"
    model_type = 'SingleGene'
    DATA_DIR = "../data"
    fold = 0
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR

    gene_name = 'BTNL3'
    train_genes = [gene_name]
    valid_genes = []
    test_genes = []    
    wandb.init(
            project = 'test',
            name = 'test',
            group = 'test',
            config = config
        )
    wandb.config.update({'fold':fold})
    wandb.config.update({'train_genes':train_genes})
    wandb.config.update({'valid_genes':valid_genes})
    wandb.config.update({'test_genes':test_genes})
    pl.seed_everything(int(wandb.config.seed), workers=True)
    torch.use_deterministic_algorithms(True)
    define_donor_paths(wandb.config,'gtex')

    #load expected data
    expected_gene_expression_df = pd.read_csv("../data/gtex_eqtl_expression_matrix/Whole_Blood.v8.normalized_expression.bed.gz",sep = '\t')
    gene_id_mapping = pd.read_csv("../data/gtex_eqtl_expression_matrix/gene_id_mapping.csv")
    expected_gene_expression_df = expected_gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')
    expected_train_donor_path = f"../data/cross_validation_folds/gtex/cv_folds/person_ids-train-fold{fold}.txt"
    regions = pd.read_csv("../data/Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv")
    #load datasets as they would be loaded during training
    train_ds, valid_ds, test_ds = load_gtex_datasets(wandb.config,train_genes,valid_genes,test_genes)

    x,y,gene,donor_id,_ = train_ds[0]

    #assert expression value is correct
    assert np.allclose(expected_gene_expression_df.loc[
        expected_gene_expression_df['Description'] == gene_name,donor_id],
        y.item())


    #assert sequence for this person is correct
    gene_regions = regions[regions['gene_name'] == gene_name]
    chrom = gene_regions['seqnames'].item()
    gene_start = int(gene_regions['gene_start'])
    start = gene_start -(49152 //2) #get 49,152bp region with TSS in the center
    end = gene_start + (49152 //2)

    consensus1_open = pysam.Fastafile(f"../data/ConsensusSeqs_SNPsOnlyUnphased/{donor_id}_consensus_H1.fa")
    consensus2_open = pysam.Fastafile(f"../data/ConsensusSeqs_SNPsOnlyUnphased/{donor_id}_consensus_H2.fa")
    seq1 = consensus1_open.fetch(chrom, start, end).upper()
    seq2 = consensus2_open.fetch(chrom, start, end).upper()
    consensus1_open.close()
    consensus2_open.close()

    expected_one_hot = (train_ds._one_hot_encode(seq1) + train_ds._one_hot_encode(seq2)) / 2
    assert np.allclose(expected_one_hot, x)
    print("All tests passed!")
def test_no_donor_overlap():
    for fold in range(10):
        train = parse_gene_files(f"../data/rosmap/person_ids-train-fold{fold}.txt")
        valid = parse_gene_files(f"../data/rosmap/person_ids-val-fold{fold}.txt")
        assert len(set(train) & set(valid)) == 0
        assert len(train) == len(set(train))
        assert len(valid) == len(set(valid))
def test_enformer_eval_rosmap():
    """
    Using Brain cortex GTEx data (but all ppl from GTEx with data, as if I had trained on rosmap and am evaluating on a totally held out cohort)
    Manually load desired dataloader and ensure the prediction you get from enformer matches what was generated in script
    """
    n_center_bins = 3
    enformer_output_bin = 4980
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    test_donor_path = os.path.join(data_dir,'cross_validation_folds/gtex/All_GTEx_ID_list.txt')
    
    expression_dir = os.path.join(data_dir,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"Brain_Cortex.v8.normalized_expression.bed.gz")
    gtex_gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gtex_gene_expression_df = gtex_gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')

    test_ds = GTExDataset(['Brain - Cortex'], ['ACOX3'], 196608 // 4, -1, test_donor_path, gtex_gene_expression_df, data_dir)
    enformer = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
    )
    enformer.cuda()
    enformer.eval()
    x,y,gene,donor,_ = test_ds[0]

    with torch.no_grad():
        y_hat = enformer(x.cuda())['human']
    y_hat = slice_enformer_pred(y_hat.unsqueeze(0),n_center_bins) #unsqueeze batch dimension because function expects one
    y_hat = y_hat[enformer_output_bin].item()

    script_result = pd.read_csv(os.path.join(cwd,f'../results/EnformerResults/FinalPaperBrainCortex/Enformer_testGTExPredictions_DonorFold_0_BrainCortexTrainTestGenes_49152bp_3CenterBins.csv'))
    script_result = script_result[(script_result['gene_name'] == gene) & (script_result['donors'] == donor)]
    script_result_y = script_result['y_true'].item()
    script_result_y_hat = script_result['model_pred'].item()
    assert script_result_y == y
    assert np.allclose(y_hat,script_result_y_hat,atol = 1e-4) #testing on diff gpu as original experiment, increase tolerance
    print("Test successful")

def test_enformer_eval_gtex():
    """
    Using blood, Manually load desired dataloader and ensure the prediction you get from enformer matches what was generated in script
    """
    n_center_bins = 3
    fold = 0
    enformer_output_bin = 4950
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    test_donor_path = os.path.join(data_dir,f'cross_validation_folds/gtex/cv_folds/person_ids-test-fold{fold}.txt') #use fold 0 here
    
    expression_dir = os.path.join(data_dir,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"Whole_Blood.v8.normalized_expression.bed.gz")
    gtex_gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gtex_gene_expression_df = gtex_gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')

    test_ds = GTExDataset(['Whole Blood'], ['BTNL3'], 196608 // 4, -1, test_donor_path, gtex_gene_expression_df, data_dir)
    enformer = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
    )
    enformer.cuda()
    enformer.eval()
    x,y,gene,donor,_ = test_ds[0]

    with torch.no_grad():
        y_hat = enformer(torch.from_numpy(x).cuda())['human']
    y_hat = slice_enformer_pred(y_hat.unsqueeze(0),n_center_bins) #unsqueeze batch dimension because function expects one
    y_hat = y_hat[enformer_output_bin].item()

    script_result = pd.read_csv(os.path.join(cwd,f'../results/EnformerResults/FinalPaperWholeBlood/Enformer_testGTExPredictions_DonorFold_{fold}_WholeBloodTrainTestGenes_49152bp_3CenterBins.csv'))
    script_result = script_result[(script_result['gene_name'] == gene) & (script_result['donors'] == donor)]
    script_result_y = script_result['y_true'].item()
    script_result_y_hat = script_result['model_pred'].item()
    assert script_result_y == y
    assert np.allclose(y_hat,script_result_y_hat,atol = 1e-4) #testing on diff gpu as original experiment, increase tolerance
    print("Test successful")
def test_performer_ism():
    """
    test that loading from checkpoint yields same prediction on validation set from the checkpoint's epoch dring training
    Then ensure you get same prediction as in the script with ref seq when you introduce a SNP
    """
    cwd = os.getcwd()
    run_id = 'bx5gnldp'
    fold = 0
    save_dir=os.path.join(cwd,f"../results/FinalPaperWholeBlood/SingleGene/BTNL3/Fold-{fold}/{run_id}")
    ckpt = 'epoch=65-step=264.ckpt'
    ckpt_epoch = int(ckpt.split('epoch=')[1].split('-')[0])
    model = load_model(ckpt,save_dir,run_id)
    model.cuda()
    model.eval()
    data_dir = os.path.join(cwd,'../data')
    valid_donor_path = os.path.join(data_dir,f'cross_validation_folds/gtex/cv_folds/person_ids-val-fold{fold}.txt') #use fold 0 here
    
    expression_dir = os.path.join(data_dir,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"Whole_Blood.v8.normalized_expression.bed.gz")
    gtex_gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gtex_gene_expression_df = gtex_gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')

    pl.seed_everything(0)
    valid_ds = GTExDataset(['Whole Blood'], ['BTNL3'], 196608 // 4, -1, valid_donor_path, gtex_gene_expression_df, data_dir)

    x,y,gene,donor,_ = valid_ds[10]
    with torch.no_grad():
        y_hat = model(torch.from_numpy(x).unsqueeze(0).cuda())
        y_hat = y_hat[:,y_hat.shape[1]//2,:].item()
    
    expected_results = pd.read_csv(os.path.join(save_dir,f"Prediction_Results_{ckpt_epoch}_in_valid_donors.csv"))
    expected_results = expected_results[(expected_results['gene'] == gene) & (expected_results['donor'] == donor)]
    expected_y_hat = expected_results['y_pred'].item()
    assert np.allclose(y_hat,expected_y_hat,atol = 1e-1) #offer lots of tolerance because this is on a diff gpu. True differences between people vary by much more than this tolerance
    print("Loading Checkpoint Test Succesful")
if __name__ == '__main__':
    test_performer_ism()