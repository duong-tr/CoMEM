#BSUB -J EKVQA               # Specify the job name
#BSUB -W 168:00                # Specify the maximum runtime in "hours:minutes"
#BSUB -o %x.%j.out            # Determine where the output will be written
#BSUB -e %x.%j.err            # The same goes for the error file
 
#BSUB -n 2                        # Specify the number of CPUS
#BSUB -M 16000                     # Specify the RAM PER CPU in MB that your job will use.
#BSUB -gpu "num=1"                # Specify the number of GPUs
#BSUB -R "span[hosts=1]"          # run on a single host
#BSUB -q batch_h100

# Receive email notifications.  Specify when to receive emails.
#BSUB -u Duong.Tran@de.bosch.com        # Specify the email address for notifications
# #BSUB -B                        # receive email when job starts
#BSUB -N                          # receive email when job ends
 
# The SLURM variables $SSD and $HDD and $SRC are not defined for LSF. Please
# work in /fs/scratch/rng_cr_bcai_dl/$USER or your home directory instead.
# Also access your data directly from /fs/scratch/rng_cr_bcai_dl . No copying of
# data onto local drives (which do not exist) is required.
 
. /fs/applications/lsf/latest/conf/profile.lsf  # THIS LINE IS MANDATORY
. /fs/applications/modules/current/init/bash    # THIS LINE IS MANDATORY
 
set -e # Exit on any error. Do not put this line before the two mandatory ones.
 
module load conda

# load latest CUDA
module load cuda/12.6.0

# To access internet within bsub script
module load proxy4server-access
source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a

# Specify your work directory here
# cd ~/
 
# Perform experiments

chmod -R +x scripts
chmod +x lsf_run.sh

./lsf_run.sh


# No longer exit on any error.
set +e
