import os

lr_list = [2e-4, 2e-5, 2e-3]
wd_list = [0.001, 0.01, 0.1]

run = 1
for lr in lr_list:
    for wd in wd_list:
        directory = "ensemble_results/Run_" + str(lr) + str(wd)
        print(f"Run {run}", "-" * 80)
        os.system(
            f"python GPT2.py --learning_rate {lr} --weight_decay {wd} --results_dir {directory}"
        )
        os.system("rm -r __pycache__")
        run += 1
