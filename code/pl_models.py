import pandas as pd
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch import Enformer
from torchmetrics.regression import PearsonCorrCoef,R2Score

def masked_mse(y_hat,y):
    """
    removes NaNs from y (for example, when someone is missing data from a particular tissue)
    """
    mask = torch.isnan(y)
    mse = torch.mean((y[~mask]-y_hat[~mask])**2)
    return mse
def get_diff_one_gene(y,y_hat):
    """
    Calculates pairwise differences between observed expression values among different people, as well as pairwise differences between predicted expression values among different people. Returns the difference between these matrices
    """
    true_differences = y.unsqueeze(1) - y
    predicted_differences = y_hat.unsqueeze(1) - y_hat
    diff = predicted_differences - true_differences
    return diff
def remove_l_tri_flatten_3d(diff):
    """
    Removes the lower triangle of `diff`, the difference between pairwise observed differences and pairwise predicted differences. Thus, it keeps only pairwise comparisons between different, unique pairs of people. This handles the case when training on multiple tissues at the same time
    and the matrix is 3D. Returns a flattened array.
    """
    # Create a mask for the upper triangle
    mask_2d = torch.triu(torch.ones(diff.shape[0:2], dtype=torch.bool), diagonal=1) #everything above diagonal is True, everything else is False. So things are True when it is the paired differences between a person and someone else
    
    mask_3d = mask_2d.unsqueeze(-1).repeat(1, 1, diff.shape[-1]) #Make this mask 3D. So the same 2D matrix is repeated along the third dimension, so the same mask is applied to each tissue in the tissue dimension
    upper_tri_3d = torch.where(mask_3d.to(diff.device), diff, torch.tensor(float('nan')).to(diff.device)) #convert False values to NaN and returns the diff values where True. Uses the 3d mask as the condition 

    # Flatten and remove NaN values
    flat = upper_tri_3d.flatten()
    return flat[~torch.isnan(flat)]

class LitModel(pl.LightningModule):
    def __init__(self,tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.pred_dict = {}
        self.target_dict = {}
        self.donor_dict = {}
        self.rank_dict = {}
        self.tissues_to_train = tissues_to_train
        self.save_dir = save_dir
        self.train_dataset = train_dataset
        self.genes_for_training = genes_for_training
        self.genes_for_valid = genes_for_valid
        self.genes_for_test = genes_for_test
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.ensure_no_gene_overlap()

    def ensure_no_gene_overlap(self):
        """
        Since it can be valuable to evaluate train genes on held out people -- to understand the extent to which the model has learned to predict loci it has seen in unseen people with unseen variants --
        It complicates the train/valid/test split of genes, because validation and test datalaoders may contain train genes (in principle). 
        
        To deal with this, I ensure here that the genes in the train dataset are only the desired train genes. During the validation and test loops, since evaluations
        may be performed on train/valid/test genes, I denote which are which using class attributes. These are accessed within the ValidMetricLogger Callback, and when this object calculates performance on these genes, 
        it labels them as train,valid, or test genes accordingly. Therefore, I triple check here, once again, that there is no overlap between these genes
        Within the `train_full_enformer function`, I also call `final_check_ensure_no_gene_or_person_leakage_in_datasets` to ensure the genes being used in each of these datasets follow these conventions.
        
        Ensuring that there is no overlap between genes for training and genes for valid and genes for test ensures genes cannot have more than 1 label, or have conflicting labels, during evaluation in ValidMetricLogger
        `final_check_ensure_no_gene_or_person_leakage_in_datasets` Will check to ensure the genes coming from these datalaoders align with these labels (while being flexible to the fact that, for example, the valid datalaoder can yield train *And* valid genes)
        """
        train_gene_set = set(self.genes_for_training)
        valid_gene_set = set(self.genes_for_valid)
        test_gene_set = set(self.genes_for_test)

        train_valid_overlap = train_gene_set & valid_gene_set
        train_test_overlap = train_gene_set & test_gene_set
        valid_test_overlap = valid_gene_set & test_gene_set

        assert len(list(train_valid_overlap)) == 0, f"There is overlap between genes in the train and valid set via the following genes {train_valid_overlap}"
        assert len(list(train_test_overlap)) == 0, f"There is overlap between genes in the train and test set via the following genes {train_test_overlap}"
        assert len(list(valid_test_overlap)) == 0, f"There is overlap between genes in the valid and test set via the following genes {valid_test_overlap}"

        assert len(self.genes_for_training) > 0, "You have no genes to train on!"

        assert len([x for x in self.train_dataset.genes_in_dataset if x not in self.genes_for_training]) == 0, "There are genes in the train set besides those desired for training!"

    def loss_fn(self,y_hat, y, alpha = 0.5): #does not need to inherit nn.Module because no trainable variables within                                                                                        
        mse = masked_mse(y_hat,y)
        diff = get_diff_one_gene(y,y_hat)
        flat = remove_l_tri_flatten_3d(diff)
        contrastive_term = torch.mean(flat**2)

        loss = (alpha * mse) + ((1 - alpha) * contrastive_term)
        return loss
    def predict_step(self, batch, batch_idx,dataloader_idx = 0):
        x = batch[0]
        y_hat = self(x)
        y_hat = y_hat[:,y_hat.shape[1]//2,:] #keep value at center of sequence. The sequence axis is removed
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y,genes,donor,dataloader_idx = batch
        y_hat = self.predict_step(batch,batch_idx)
        loss = self.loss_fn(y_hat, y) 
        self.log('train_loss',loss,batch_size = y.shape[0], on_step = False, on_epoch = True) #accumulates loss over the epoch and only logs the average at the end, to reduce logging overhead
        return {'loss': loss}
    
    def save_eval_results(self,y_hat,y,donor,rank,gene_name):
        """
        To log predictions and true values for each donor during validation/test loop, separately in each tissue (if training multiple tissues)
        """
        if gene_name not in self.pred_dict:
            self.pred_dict[gene_name] = {tissue:[] for tissue in self.tissues_to_train}
            self.target_dict[gene_name] = {tissue:[] for tissue in self.tissues_to_train}
            self.donor_dict[gene_name] = {tissue:[] for tissue in self.tissues_to_train} 
            self.rank_dict[gene_name] = {tissue:[] for tissue in self.tissues_to_train} 

        tissue_indices_without_data = torch.isnan(y).nonzero(as_tuple=True)[1] #NaNs exist in tissues where donor is missing data. Expected validation batch size is 1, so this finds tissues where this individual is missing data, and results for this person won't be saved for this tissue
        
        for tissue_idx,tissue in enumerate(self.tissues_to_train):
            if tissue_idx not in tissue_indices_without_data:
                self.pred_dict[gene_name][tissue].append(y_hat[:,tissue_idx])
                self.target_dict[gene_name][tissue].append(y[:,tissue_idx])
                self.donor_dict[gene_name][tissue].append(donor)
                self.rank_dict[gene_name][tissue].append(rank)

    def validation_step(self, batch, batch_idx,dataloader_idx = 0):
        x, y,gene_name,donor,_ = batch
        gene_name = gene_name[0] #expected batch size is 1 for validation loop
        donor = donor[0]
        rank = self.trainer.global_rank
        y_hat = self.predict_step(batch,batch_idx)
        self.save_eval_results(y_hat,y,donor,rank,gene_name) #results for each person are stored and then loss/r2/pcc will be computed at the end of the epoch using all individuals via MetricLogger Callback
    def test_step(self, batch, batch_idx,dataloader_idx = 0):
        x, y,gene_name,donor,_ = batch
        gene_name = gene_name[0] #expected batch size is 1 for test loop
        donor = donor[0]
        rank = self.trainer.global_rank
        y_hat = self.predict_step(batch,batch_idx)
        self.save_eval_results(y_hat,y,donor,rank,gene_name)
    def on_train_epoch_end(self):
        self.train_dataset.shuffle_and_define_epoch() #shuffle dataset. Ensures this occurs on the main process even if num_workers > 0
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        return {
            'optimizer': optimizer
        }
    
class LitModelHeadAdapterWrapper(LitModel):
    def __init__(self, tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test):
        super().__init__(tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test)

        enformer = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )

        self.model = HeadAdapterWrapper(
            enformer = enformer,
            num_tracks = len(self.tissues_to_train),
            post_transformer_embed = False, # important to keep False
            output_activation = nn.Identity()
        )

    def forward(self, x):
        return self.model(x, freeze_enformer = False)

class LitModelEvalBN(LitModel):
    """
    Overwrite LitModel to freeze batchnorm updates. Can be useful if not useing pytorch-enformer's HeadAdapterWrapper, which handles this for you
    """
    def __init__(self, model,tissues_to_train,save_dir,valid_metrics_save_freq,train_dataset,target_learning_rate,loss_func_str,alpha,genes_for_training,genes_for_valid,loss_funcs,genes_for_test):
        super().__init__(model,tissues_to_train,save_dir,valid_metrics_save_freq,train_dataset,target_learning_rate,loss_func_str,alpha,genes_for_training,genes_for_valid,loss_funcs,genes_for_test)
        self.freeze_batchnorm(self.model)

    def freeze_batchnorm(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d) or \
               isinstance(module, torch.nn.BatchNorm2d) or \
               isinstance(module, torch.nn.BatchNorm3d):
                module.eval()

    def train(self, mode=True): #overwrite model.train() to keep batchnorm layers in eval mode
        super().train(mode)  # Sets the model to training mode like usual
        self.freeze_batchnorm(self.model)  # Revert BatchNorm layers to eval mode
        return self





class MetricLogger(pl.Callback):
    """
    A callback intended to log and save raw predictions and summary statistics (R2, PCC) from validation/test loops
    """
    def __init__(self):
        super().__init__()
        self.metrics_history = {'epoch': [],'pearsonr':[], 'r2':[],'gene_name': [],'tissue': [],'per_gene_tissue_val_loss': [],'gene_split':[],'donor_split': []}
    def log_predictions(self,trainer,pl_module,donor_split):
        # pred_df = pd.DataFrame(pl_module.pred_dict).apply(pd.Series.explode)
        pred_df = pd.DataFrame(pl_module.pred_dict)
        pred_df = pred_df.reset_index(names = 'tissue').melt(id_vars = ['tissue'],var_name = 'gene',value_name = 'y_pred')
        pred_df['y_pred'] = pred_df['y_pred'].apply(lambda y_pred: [x.item() for x in y_pred]) #y_pred is a list of torch tensors. convert it to a list of floats
        pred_df = pred_df.explode('y_pred') #explode the list of floats so they each get their own row and the list is removed
        # target_df = pd.DataFrame(pl_module.target_dict).apply(pd.Series.explode)
        target_df = pd.DataFrame(pl_module.target_dict)
        target_df = target_df.reset_index(names = 'tissue').melt(id_vars = ['tissue'],var_name = 'gene',value_name = 'y_true')
        target_df['y_true'] = target_df['y_true'].apply(lambda y_true: [x.item() for x in y_true]) #y_pred is a list of torch tensors. convert it to a list of floats
        target_df = target_df.explode('y_true')
        # donor_df = pd.DataFrame(pl_module.donor_dict).apply(pd.Series.explode)
        donor_df = pd.DataFrame(pl_module.donor_dict)
        donor_df = donor_df.reset_index(names = 'tissue').melt(id_vars = ['tissue'],var_name = 'gene',value_name = 'donor')
        donor_df = donor_df.explode('donor')

        rank_df = pd.DataFrame(pl_module.rank_dict)
        rank_df = rank_df.reset_index(names = 'tissue').melt(id_vars = ['tissue'],var_name = 'gene',value_name = 'rank')
        rank_df = rank_df.explode('rank')
  
        df = pd.concat([pred_df.reset_index(),target_df[['y_true']].reset_index(),donor_df[['donor']].reset_index(),rank_df[['rank']].reset_index()],axis = 1)
        df['end_of_epoch'] = self.epoch
        # trainer.logger.experiment.log({'PredictionResults':wandb.Table(dataframe = df)})
        df.to_csv(os.path.join(pl_module.save_dir,f"Prediction_Results_{self.epoch}_in_{donor_split}_donors.csv"), index=False)
    def add_to_metrics_history(self,gene_name,tissue,pl_module,gene_split,pearson,r2_score,donor_split):
        """
        Indexes dictionaries of predictions and true values for a given gene and records summary statistics
        """
        loss_fn = pl_module.loss_fn #if the train loss func is a multi gene one, it doesn't work here because this code runs one gene at a time. The valid loss func returns a single gene loss func
        all_predictions = torch.cat(pl_module.pred_dict[gene_name][tissue])
        all_targets = torch.cat(pl_module.target_dict[gene_name][tissue])

        gene_pearsonr = pearson(all_predictions,all_targets)
        gene_r2 = r2_score(all_predictions,all_targets)

        loss_val = loss_fn(all_predictions.unsqueeze(1),all_targets.unsqueeze(1)).cpu().numpy() #unsqueeze because the values per tissue were returned and have shape [batch_size]. Add a tissue dimension, which the loss function expects
        self.metrics_history['pearsonr'].append(gene_pearsonr.detach().cpu().numpy()) 
        self.metrics_history['r2'].append(gene_r2.detach().cpu().numpy()) 
        self.metrics_history['tissue'].append(tissue)
        self.metrics_history['gene_name'].append(gene_name)
        self.metrics_history['epoch'].append(self.epoch)
        self.metrics_history['per_gene_tissue_val_loss'].append(loss_val)
        self.metrics_history['gene_split'].append(gene_split)
        self.metrics_history['donor_split'].append(donor_split)
    
    def log_per_gene_per_tissue_metrics(self,trainer,pl_module,donor_split):
        metrics_history = pd.DataFrame(self.metrics_history)
        df = metrics_history[metrics_history['donor_split'] == donor_split].copy()
        name = f"CrossIndivMetrics_{donor_split}_donors_Epoch{self.epoch}.csv"
        df.to_csv(os.path.join(pl_module.save_dir,name), index=False)
        
        assert len(df['donor_split'].unique()) == 1 & (df['donor_split'].unique().item() == donor_split), f"You have data from {df['donor_split'].unique()} donors but should only have {donor_split}!"
        for gene_split in list(df['gene_split'].unique()):
            #save results across all tissues for this epoch
            mean_epoch_corr = df[df['gene_split'] == gene_split]['pearsonr'].mean()
            mean_epoch_r2 = df[df['gene_split'] == gene_split]['r2'].mean()
            mean_epoch_loss = df[df['gene_split'] == gene_split]['per_gene_tissue_val_loss'].mean()
            epoch_dict = { 
                f'mean_pearsonr_across_{gene_split}_genes_across_{donor_split}_donors':mean_epoch_corr.item(),
                f'mean_r2_across_{gene_split}_genes_across_{donor_split}_donors':mean_epoch_r2.item(),
                f'mean_loss_{gene_split}_genes_across_{donor_split}_donors':mean_epoch_loss.item()
            }
            if trainer.num_devices > 1: #sync across multiple GPUs, if applicable
                pl_module.log_dict(epoch_dict,sync_dist = True)
            else:
                pl_module.log_dict(epoch_dict)
    
    def add_all_genes_to_metrics_history(self,pl_module,donor_split):
        pearson = PearsonCorrCoef(num_outputs=1).to(pl_module.device)
        r2_score = R2Score(num_outputs=1).to(pl_module.device)
        for gene_name in pl_module.pred_dict.keys():
            if gene_name in pl_module.genes_for_training:
                gene_split = 'train'
            elif gene_name in pl_module.genes_for_valid:
                gene_split = 'valid'
            elif gene_name in pl_module.genes_for_test:
                gene_split = 'test'
            else:
                raise Exception(f"Gene {gene_name} not in desired train set, valid set, or test set")
            for tissue_idx,tissue in enumerate(pl_module.tissues_to_train):  # loop through each tissue and calculate pearsonr
                self.add_to_metrics_history(gene_name,tissue,pl_module,gene_split,pearson,r2_score,donor_split)


    def get_epoch(self,trainer,pl_module):
        if pl_module.global_step == 0: #when performing validation loop before training, no steps will have occured but epoch is defined as 0. It will be overwritte after the 0th training epoch. Define as -1 to save these values.
            epoch = -1
        else:
            epoch = pl_module.current_epoch
        self.epoch = epoch

    def log_and_save_eval(self,trainer,pl_module,donor_split):
        self.get_epoch(trainer,pl_module) #define self.epoch
        self.add_all_genes_to_metrics_history(pl_module,donor_split)
        self.log_predictions(trainer,pl_module,donor_split) #log y_pred and y_true for each donor and gene being evaluated
        self.log_per_gene_per_tissue_metrics(trainer,pl_module,donor_split)

        # Clear predictions and targets for the next epoch
        pl_module.pred_dict = {}
        pl_module.target_dict = {}
        pl_module.donor_dict = {}
        self.metrics_history = {'epoch': [],'pearsonr':[], 'r2':[],'gene_name': [],'tissue': [],'per_gene_tissue_val_loss': [],'gene_split':[],'donor_split': []}
    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_and_save_eval(trainer,pl_module,'valid')
    def on_test_epoch_end(self,trainer,pl_module):
        self.log_and_save_eval(trainer,pl_module,'test')   
