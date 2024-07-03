import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
from train_gtex import *
from eval_enformer_gtex import slice_enformer_pred
from ism_performer import tss_centered_sequences, get_window_around_TSS,load_model, one_hot_encode, IsmDataset,LitModelPerformerISM
from enformer_pytorch import Enformer

torch.use_deterministic_algorithms(True)
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
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    for fold in range(10):
        #test rosmap
        train = parse_gene_files(os.path.join(data_dir,f"cross_validation_folds/rosmap/person_ids-train-fold{fold}.txt"))
        valid = parse_gene_files(os.path.join(data_dir,f"cross_validation_folds/rosmap/person_ids-val-fold{fold}.txt"))
        assert len(set(train) & set(valid)) == 0
        assert len(train) == len(set(train))
        assert len(valid) == len(set(valid))

        #test gtex
        train = parse_gene_files(os.path.join(data_dir,f"cross_validation_folds/gtex/cv_folds/person_ids-train-fold{fold}.txt"))
        valid = parse_gene_files(os.path.join(data_dir,f"cross_validation_folds/gtex/cv_folds/person_ids-val-fold{fold}.txt"))
        test = parse_gene_files(os.path.join(data_dir,f"cross_validation_folds/gtex/cv_folds/person_ids-test-fold{fold}.txt"))
        assert len(set(train) & set(valid)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(valid) & set(test)) == 0
        assert len(train) == len(set(train))
        assert len(valid) == len(set(valid))
        assert len(test) == len(set(test))
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
        y_hat = enformer(torch.from_numpy(x).cuda())['human']
    y_hat = slice_enformer_pred(y_hat.unsqueeze(0),n_center_bins) #unsqueeze batch dimension because function expects one
    y_hat = y_hat[enformer_output_bin].item()

    script_result = pd.read_csv(os.path.join(cwd,f'../results/EnformerResults/FinalPaperBrainCortex/Enformer_testGTExPredictions_DonorFold_0_BrainCortexTrainTestGenes_49152bp_3CenterBins.csv'))
    script_result = script_result[(script_result['gene_name'] == gene) & (script_result['donors'] == donor)]
    script_result_y = script_result['y_true'].item()
    script_result_y_hat = script_result['model_pred'].item()
    assert script_result_y == y
    assert np.allclose(y_hat,script_result_y_hat,rtol = 1e-3) #testing on diff gpu as original experiment with bf-16 mixed precision, increase tolerance
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
def test_performer_checkpoint_and_ism():
    """
    test that loading from checkpoint yields same prediction on validation set from the checkpoint's epoch dring training
    Then ensure you get same prediction as in the script with ref seq when you introduce a SNP
    """
    cwd = os.getcwd()
    desired_seq_len = 196608 // 4
    run_id = '1jmnxw51'
    fold = 0

    
    save_dir=os.path.join(cwd,f"../results/FinalPaperWholeBlood/MultiGene/300_train_genes/Fold-{fold}/{run_id}")
    ckpt = 'epoch=16-step=20468.ckpt'
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
    valid_ds = GTExDataset(['Whole Blood'], ['BTNL3'], desired_seq_len, -1, valid_donor_path, gtex_gene_expression_df, data_dir)

    x,y,gene,donor,_ = valid_ds[10]
    with torch.no_grad():
        y_hat = model(torch.from_numpy(x).unsqueeze(0).cuda())
        y_hat = y_hat[:,y_hat.shape[1]//2,:].item()
    
    expected_results = pd.read_csv(os.path.join(save_dir,f"Prediction_Results_{ckpt_epoch}_in_valid_donors.csv"))
    expected_results = expected_results[(expected_results['gene'] == gene) & (expected_results['donor'] == donor)]
    expected_y_hat = expected_results['y_pred'].item()
    assert np.allclose(y_hat,expected_y_hat,atol = 1e-1) #offer lots of tolerance because this is on a diff gpu. True differences between people vary by much more than this tolerance
    print("Loading Checkpoint Test Succesful")

    metadata = gen_ism_data()
    with torch.no_grad():
        ref_pred = model(torch.from_numpy(metadata['inputs']['ref']).unsqueeze(0).cuda())
        ref_pred = ref_pred[:,ref_pred.shape[1]//2,:].item()
        
        alt_pred = model(torch.from_numpy(metadata['inputs']['alt']).unsqueeze(0).cuda())
        alt_pred = alt_pred[:,alt_pred.shape[1]//2,:].item()

    experimental_results = pd.read_csv(os.path.join(cwd,f'expected_results/Whole_BloodModels/MultiGene/{run_id}/BTNL3_{run_id}_model_ISM_{desired_seq_len}bp.csv'))
    variant_results = experimental_results[experimental_results['pos0'] == metadata['metadata']['pos']]
    ref_results = variant_results['ref_pred'].item()
    alt_results = variant_results['alt_pred'].item()
    assert np.allclose(ref_results,ref_pred,rtol = 1e-2) #offer extra tolerance because the experiment was run on a different gpu with bf-16 mixed precision
    assert np.allclose(alt_results,alt_pred,rtol = 1e-2)
    print("Performer ISM Succeeded")

    #repeat using trainer and dataloader, no tolerance needed because precision is the same
    it = gen_ism_data(return_it = True)
    dataset = IsmDataset(it,length = 1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    lit_model = LitModelPerformerISM(model, run_id, ckpt)
    trainer = pl.Trainer(precision='bf16-mixed',
                    num_sanity_val_steps = 0, #check all validation data before starting to train
                    deterministic = True)  
    trainer.predict(lit_model,dataloader)
    expected_alt_pred = lit_model.results_df['alt_pred'].item()
    expected_ref_pred = lit_model.results_df['ref_pred'].item()
    assert np.allclose(alt_results,expected_alt_pred)
    assert np.allclose(ref_results,expected_ref_pred)

def gen_ism_data(return_it = False):
    desired_seq_len = 196608 // 4
    variant_dict = {'region': ['chr5:180964268-181013420'],
                    'chrom': ['chr5'],
                    'pos0': [180993126],
                    'pos1': [180993127],
                    'ref': ['G'],
                    'alt': ['A'],
                    'AF': [0.00119332],
                    'gene_name': ['BTNL3']}
    variant_df = pd.DataFrame(variant_dict)
    it = tss_centered_sequences(variant_df,desired_seq_len)
    if return_it:
        return it
    else:
        for metadata in it:
            break
        return metadata
def test_tss_centered_sequences():
    """
    Tests generator that returns reference and alternate allele sequences for ISM
    """
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,"../data")
    gene = 'BTNL3'
    desired_seq_len = 196608 // 4
    ref_seq_open = pysam.Fastafile(os.path.join(DATA_DIR,"hg38_genome.fa"))
    enformer_regions = pd.read_csv(os.path.join(DATA_DIR,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    gene_info = enformer_regions[enformer_regions['gene_name'] == gene]
    seq_window = desired_seq_len // 2 #the sequence used for ism has length seq_window * 2 because it is before and after the TSS. Divide by 2 to get a sequence whose length is desired_seq_len
    region = get_window_around_TSS(seq_window,gene_info)
    region_chr = region.split(':')[0]
    region_start = int(region.split(':')[1].split('-')[0])
    region_end = int(region.split(':')[1].split('-')[1])
    assert region_end - region_start == desired_seq_len
    ref_seq = ref_seq_open.fetch(region_chr, region_start, region_end).upper()


    #make up variant_df, check that you get what you expect
    variant_dict = {'region': ['chr5:180964268-181013420'],
                    'chrom': ['chr5'],
                    'pos0': [180993126],
                    'pos1': [180993127],
                    'ref': ['G'],
                    'alt': ['A'],
                    'AF': [0.00119332],
                    'gene_name': ['BTNL3']}
    variant_df = pd.DataFrame(variant_dict)
    it = tss_centered_sequences(variant_df,desired_seq_len)

    index = variant_df['pos0'].item() - region_start
    assert ref_seq[index] == variant_df['ref'].item(), "nucleotide in reference genome should be identical to the reference allele of the current SNP at the current position"
    alt_seq = ref_seq[:index] + variant_df['alt'].item() + ref_seq[index + 1:] # put alt allele in its correct position and ref seq around it
    
    for metadata in it:
        break
    assert np.allclose(metadata['inputs']['ref'],one_hot_encode(ref_seq))
    assert np.allclose(metadata['inputs']['alt'],one_hot_encode(alt_seq))

def test_enformer_ism():
    cwd = os.getcwd()    
    metadata = gen_ism_data()
    model = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        ) 
    model.eval()
    model.cuda() 
    enformer_output_dim = 4950
    with torch.no_grad():
        ref_pred = model(torch.from_numpy(metadata['inputs']['ref']).unsqueeze(0).cuda())['human']
        alt_pred = model(torch.from_numpy(metadata['inputs']['alt']).unsqueeze(0).cuda())['human']

        ref_pred = slice_enformer_pred(ref_pred,3)
        alt_pred = slice_enformer_pred(alt_pred,3)

        ref_pred = ref_pred[enformer_output_dim].item()
        alt_pred = alt_pred[enformer_output_dim].item()
    experimental_results = pd.read_csv(os.path.join(cwd,'../code/results/EnformerISM/BTNL3_Enformer_model_ISM_49152bp_Brain - Cortex,Whole Blood_3CenterBins.csv'))
    variant_results = experimental_results[experimental_results['pos0'] == metadata['metadata']['pos']]
    variant_results = variant_results[variant_results['enformer_output_dim'] == enformer_output_dim]
    ref_results = variant_results['ref_pred'].item()
    alt_results = variant_results['alt_pred'].item()
    assert np.allclose(ref_results,ref_pred,atol = 1e-5)
    assert np.allclose(alt_results,alt_pred,atol = 1e-5)

def test_checkpoint_during_training():
    """
    After training, evaluat eon the validation set after restarting from the best checkpoint. These results should be identical to the validation performance during training if checkpointing worked out
    """
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,'../data')
    config_path="/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/configs/blood_config.yaml"
    model_type='SingleGene'
    fold=0

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR

    train_genes = ['WBP4']
    valid_genes = []
    test_genes = []

    wandb.init(
        project = 'test',
        name = config['experiment_name'] + f'_Fold-{fold}_' + 'test',
        group = config['experiment_name'],
        config = config
    )
    wandb.config.update({'fold':fold})
    wandb.config.update({'train_genes':train_genes})
    wandb.config.update({'valid_genes':valid_genes})
    wandb.config.update({'test_genes':test_genes})
    wandb.config.update({'save_dir' : os.path.join(cwd,"expected_results/test_checkpoint")})
    wandb.config.update({'train_batch_size' : 8},allow_val_change=True)
    wandb.config.update({'max_epochs' : 15},allow_val_change=True)
    pl.seed_everything(int(wandb.config.seed), workers=True)
    torch.use_deterministic_algorithms(True)
    
    config = wandb.config

    ensure_no_gene_overlap(train_genes,valid_genes,test_genes)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    model = LitModelHeadAdapterWrapper(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes
    )
    trainer = load_trainer(config)
    trainer.fit(model = model,
                train_dataloaders = DataLoader(train_ds,batch_size = config.train_batch_size),
                val_dataloaders = DataLoader(valid_ds, batch_size = 1) #code for logging and storing validation/test results expects batch size of 1 for these 
                ) 
    trainer.validate(model, DataLoader(valid_ds,batch_size = 1), ckpt_path = 'best')
    trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = 'best')
    ckpt_epoch = os.listdir(os.path.join(config.save_dir,'checkpoints'))[0]
    ckpt_epoch = int(ckpt_epoch.split('epoch=')[1].split('-')[0])
    valid_during_training = pd.read_csv(os.path.join(config.save_dir,f"CrossIndivMetrics_valid_donors_Epoch{ckpt_epoch}.csv"))
    valid_after_rewind = pd.read_csv(os.path.join(config.save_dir,f"CrossIndivMetrics_valid_donors_Epoch{trainer.current_epoch}.csv"))

    assert valid_during_training.r2.item() == valid_after_rewind.r2.item()
    wandb.finish()

        
    




if __name__ == '__main__':
    test_checkpoint_during_training()
    # test_performer_checkpoint_and_ism()
    # test_enformer_ism()
    # test_tss_centered_sequences()
    # test_enformer_eval_gtex()
    # test_enformer_eval_rosmap()
    # test_no_donor_overlap()
    # test_train_dataloader()
    print("All tests passed!")