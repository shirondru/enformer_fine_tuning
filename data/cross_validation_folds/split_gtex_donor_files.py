import os
def split_file():
    """
    To split up a donor file into chunks to test effect of sample size on training
    """
    current_dir = os.path.dirname(__file__)

    for fold in range(10):
        original_file_path = os.path.join(current_dir,f'gtex/cv_folds/person_ids-train-fold{fold}.txt')

        # Read all lines from the original file containing donor IDs on different lines
        with open(original_file_path, 'r') as file:
            lines = file.readlines()
        n_original_donors = len(lines)
        
        outdir = os.path.join(current_dir,f'gtex/downsampled_train_files')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # Each file contains an increment of 25% of the total dononrs
        for i in range(1, 5):
            n_donors_to_use = (n_original_donors * i) // 4
            new_file_path = os.path.join(outdir,f'person_ids-train-fold{fold}_percentage{i}_4.txt')
            with open(new_file_path, 'w') as new_file:
                new_file.writelines(lines[:n_donors_to_use])

def main():
    split_file()

if __name__ == "__main__":
    main()