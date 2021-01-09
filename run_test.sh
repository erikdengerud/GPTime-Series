#!/bin/sh
chmod u+x slurm_jobs/test/Y-smape_loss-naive-32.yml.slurm
RES0=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-32.yml.slurm)
echo "$RES0, slurm_jobs/test/Y-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-smape_loss-naive-64.yml.slurm
RES1=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-64.yml.slurm)
echo "$RES1, slurm_jobs/test/Y-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-smape_loss-naive-128.yml.slurm
RES2=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-128.yml.slurm)
echo "$RES2, slurm_jobs/test/Y-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-smape_loss-naive-256.yml.slurm
RES3=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-256.yml.slurm)
echo "$RES3, slurm_jobs/test/Y-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-smape_loss-naive-512.yml.slurm
RES4=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-512.yml.slurm)
echo "$RES4, slurm_jobs/test/Y-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Y-smape_loss-naive-1024.yml.slurm
RES5=$(sbatch --parsable slurm_jobs/test/Y-smape_loss-naive-1024.yml.slurm)
echo "$RES5, slurm_jobs/test/Y-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mase_loss-naive-32.yml.slurm
RES6=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-32.yml.slurm)
echo "$RES6, slurm_jobs/test/Y-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mase_loss-naive-64.yml.slurm
RES7=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-64.yml.slurm)
echo "$RES7, slurm_jobs/test/Y-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mase_loss-naive-128.yml.slurm
RES8=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-128.yml.slurm)
echo "$RES8, slurm_jobs/test/Y-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mase_loss-naive-256.yml.slurm
RES9=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-256.yml.slurm)
echo "$RES9, slurm_jobs/test/Y-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Y-mase_loss-naive-512.yml.slurm
RES10=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-512.yml.slurm)
echo "$RES10, slurm_jobs/test/Y-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mase_loss-naive-1024.yml.slurm
RES11=$(sbatch --parsable slurm_jobs/test/Y-mase_loss-naive-1024.yml.slurm)
echo "$RES11, slurm_jobs/test/Y-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mape_loss-naive-32.yml.slurm
RES12=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-32.yml.slurm)
echo "$RES12, slurm_jobs/test/Y-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mape_loss-naive-64.yml.slurm
RES13=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-64.yml.slurm)
echo "$RES13, slurm_jobs/test/Y-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mape_loss-naive-128.yml.slurm
RES14=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-128.yml.slurm)
echo "$RES14, slurm_jobs/test/Y-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Y-mape_loss-naive-256.yml.slurm
RES15=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-256.yml.slurm)
echo "$RES15, slurm_jobs/test/Y-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mape_loss-naive-512.yml.slurm
RES16=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-512.yml.slurm)
echo "$RES16, slurm_jobs/test/Y-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Y-mape_loss-naive-1024.yml.slurm
RES17=$(sbatch --parsable slurm_jobs/test/Y-mape_loss-naive-1024.yml.slurm)
echo "$RES17, slurm_jobs/test/Y-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-smape_loss-naive-32.yml.slurm
RES18=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-32.yml.slurm)
echo "$RES18, slurm_jobs/test/Q-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-smape_loss-naive-64.yml.slurm
RES19=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-64.yml.slurm)
echo "$RES19, slurm_jobs/test/Q-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Q-smape_loss-naive-128.yml.slurm
RES20=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-128.yml.slurm)
echo "$RES20, slurm_jobs/test/Q-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-smape_loss-naive-256.yml.slurm
RES21=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-256.yml.slurm)
echo "$RES21, slurm_jobs/test/Q-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-smape_loss-naive-512.yml.slurm
RES22=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-512.yml.slurm)
echo "$RES22, slurm_jobs/test/Q-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-smape_loss-naive-1024.yml.slurm
RES23=$(sbatch --parsable slurm_jobs/test/Q-smape_loss-naive-1024.yml.slurm)
echo "$RES23, slurm_jobs/test/Q-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mase_loss-naive-32.yml.slurm
RES24=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-32.yml.slurm)
echo "$RES24, slurm_jobs/test/Q-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Q-mase_loss-naive-64.yml.slurm
RES25=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-64.yml.slurm)
echo "$RES25, slurm_jobs/test/Q-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mase_loss-naive-128.yml.slurm
RES26=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-128.yml.slurm)
echo "$RES26, slurm_jobs/test/Q-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mase_loss-naive-256.yml.slurm
RES27=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-256.yml.slurm)
echo "$RES27, slurm_jobs/test/Q-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mase_loss-naive-512.yml.slurm
RES28=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-512.yml.slurm)
echo "$RES28, slurm_jobs/test/Q-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mase_loss-naive-1024.yml.slurm
RES29=$(sbatch --parsable slurm_jobs/test/Q-mase_loss-naive-1024.yml.slurm)
echo "$RES29, slurm_jobs/test/Q-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Q-mape_loss-naive-32.yml.slurm
RES30=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-32.yml.slurm)
echo "$RES30, slurm_jobs/test/Q-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mape_loss-naive-64.yml.slurm
RES31=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-64.yml.slurm)
echo "$RES31, slurm_jobs/test/Q-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mape_loss-naive-128.yml.slurm
RES32=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-128.yml.slurm)
echo "$RES32, slurm_jobs/test/Q-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mape_loss-naive-256.yml.slurm
RES33=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-256.yml.slurm)
echo "$RES33, slurm_jobs/test/Q-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/Q-mape_loss-naive-512.yml.slurm
RES34=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-512.yml.slurm)
echo "$RES34, slurm_jobs/test/Q-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/Q-mape_loss-naive-1024.yml.slurm
RES35=$(sbatch --parsable slurm_jobs/test/Q-mape_loss-naive-1024.yml.slurm)
echo "$RES35, slurm_jobs/test/Q-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-smape_loss-naive-32.yml.slurm
RES36=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-32.yml.slurm)
echo "$RES36, slurm_jobs/test/M-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-smape_loss-naive-64.yml.slurm
RES37=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-64.yml.slurm)
echo "$RES37, slurm_jobs/test/M-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-smape_loss-naive-128.yml.slurm
RES38=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-128.yml.slurm)
echo "$RES38, slurm_jobs/test/M-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-smape_loss-naive-256.yml.slurm
RES39=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-256.yml.slurm)
echo "$RES39, slurm_jobs/test/M-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/M-smape_loss-naive-512.yml.slurm
RES40=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-512.yml.slurm)
echo "$RES40, slurm_jobs/test/M-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-smape_loss-naive-1024.yml.slurm
RES41=$(sbatch --parsable slurm_jobs/test/M-smape_loss-naive-1024.yml.slurm)
echo "$RES41, slurm_jobs/test/M-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mase_loss-naive-32.yml.slurm
RES42=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-32.yml.slurm)
echo "$RES42, slurm_jobs/test/M-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mase_loss-naive-64.yml.slurm
RES43=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-64.yml.slurm)
echo "$RES43, slurm_jobs/test/M-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mase_loss-naive-128.yml.slurm
RES44=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-128.yml.slurm)
echo "$RES44, slurm_jobs/test/M-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/M-mase_loss-naive-256.yml.slurm
RES45=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-256.yml.slurm)
echo "$RES45, slurm_jobs/test/M-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mase_loss-naive-512.yml.slurm
RES46=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-512.yml.slurm)
echo "$RES46, slurm_jobs/test/M-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mase_loss-naive-1024.yml.slurm
RES47=$(sbatch --parsable slurm_jobs/test/M-mase_loss-naive-1024.yml.slurm)
echo "$RES47, slurm_jobs/test/M-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mape_loss-naive-32.yml.slurm
RES48=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-32.yml.slurm)
echo "$RES48, slurm_jobs/test/M-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mape_loss-naive-64.yml.slurm
RES49=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-64.yml.slurm)
echo "$RES49, slurm_jobs/test/M-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/M-mape_loss-naive-128.yml.slurm
RES50=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-128.yml.slurm)
echo "$RES50, slurm_jobs/test/M-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mape_loss-naive-256.yml.slurm
RES51=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-256.yml.slurm)
echo "$RES51, slurm_jobs/test/M-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mape_loss-naive-512.yml.slurm
RES52=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-512.yml.slurm)
echo "$RES52, slurm_jobs/test/M-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/M-mape_loss-naive-1024.yml.slurm
RES53=$(sbatch --parsable slurm_jobs/test/M-mape_loss-naive-1024.yml.slurm)
echo "$RES53, slurm_jobs/test/M-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-smape_loss-naive-32.yml.slurm
RES54=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-32.yml.slurm)
echo "$RES54, slurm_jobs/test/W-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/W-smape_loss-naive-64.yml.slurm
RES55=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-64.yml.slurm)
echo "$RES55, slurm_jobs/test/W-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-smape_loss-naive-128.yml.slurm
RES56=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-128.yml.slurm)
echo "$RES56, slurm_jobs/test/W-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-smape_loss-naive-256.yml.slurm
RES57=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-256.yml.slurm)
echo "$RES57, slurm_jobs/test/W-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-smape_loss-naive-512.yml.slurm
RES58=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-512.yml.slurm)
echo "$RES58, slurm_jobs/test/W-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-smape_loss-naive-1024.yml.slurm
RES59=$(sbatch --parsable slurm_jobs/test/W-smape_loss-naive-1024.yml.slurm)
echo "$RES59, slurm_jobs/test/W-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/W-mase_loss-naive-32.yml.slurm
RES60=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-32.yml.slurm)
echo "$RES60, slurm_jobs/test/W-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mase_loss-naive-64.yml.slurm
RES61=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-64.yml.slurm)
echo "$RES61, slurm_jobs/test/W-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mase_loss-naive-128.yml.slurm
RES62=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-128.yml.slurm)
echo "$RES62, slurm_jobs/test/W-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mase_loss-naive-256.yml.slurm
RES63=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-256.yml.slurm)
echo "$RES63, slurm_jobs/test/W-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mase_loss-naive-512.yml.slurm
RES64=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-512.yml.slurm)
echo "$RES64, slurm_jobs/test/W-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/W-mase_loss-naive-1024.yml.slurm
RES65=$(sbatch --parsable slurm_jobs/test/W-mase_loss-naive-1024.yml.slurm)
echo "$RES65, slurm_jobs/test/W-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mape_loss-naive-32.yml.slurm
RES66=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-32.yml.slurm)
echo "$RES66, slurm_jobs/test/W-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mape_loss-naive-64.yml.slurm
RES67=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-64.yml.slurm)
echo "$RES67, slurm_jobs/test/W-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mape_loss-naive-128.yml.slurm
RES68=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-128.yml.slurm)
echo "$RES68, slurm_jobs/test/W-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mape_loss-naive-256.yml.slurm
RES69=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-256.yml.slurm)
echo "$RES69, slurm_jobs/test/W-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/W-mape_loss-naive-512.yml.slurm
RES70=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-512.yml.slurm)
echo "$RES70, slurm_jobs/test/W-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/W-mape_loss-naive-1024.yml.slurm
RES71=$(sbatch --parsable slurm_jobs/test/W-mape_loss-naive-1024.yml.slurm)
echo "$RES71, slurm_jobs/test/W-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-smape_loss-naive-32.yml.slurm
RES72=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-32.yml.slurm)
echo "$RES72, slurm_jobs/test/D-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-smape_loss-naive-64.yml.slurm
RES73=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-64.yml.slurm)
echo "$RES73, slurm_jobs/test/D-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-smape_loss-naive-128.yml.slurm
RES74=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-128.yml.slurm)
echo "$RES74, slurm_jobs/test/D-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/D-smape_loss-naive-256.yml.slurm
RES75=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-256.yml.slurm)
echo "$RES75, slurm_jobs/test/D-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-smape_loss-naive-512.yml.slurm
RES76=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-512.yml.slurm)
echo "$RES76, slurm_jobs/test/D-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-smape_loss-naive-1024.yml.slurm
RES77=$(sbatch --parsable slurm_jobs/test/D-smape_loss-naive-1024.yml.slurm)
echo "$RES77, slurm_jobs/test/D-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mase_loss-naive-32.yml.slurm
RES78=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-32.yml.slurm)
echo "$RES78, slurm_jobs/test/D-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mase_loss-naive-64.yml.slurm
RES79=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-64.yml.slurm)
echo "$RES79, slurm_jobs/test/D-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/D-mase_loss-naive-128.yml.slurm
RES80=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-128.yml.slurm)
echo "$RES80, slurm_jobs/test/D-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mase_loss-naive-256.yml.slurm
RES81=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-256.yml.slurm)
echo "$RES81, slurm_jobs/test/D-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mase_loss-naive-512.yml.slurm
RES82=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-512.yml.slurm)
echo "$RES82, slurm_jobs/test/D-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mase_loss-naive-1024.yml.slurm
RES83=$(sbatch --parsable slurm_jobs/test/D-mase_loss-naive-1024.yml.slurm)
echo "$RES83, slurm_jobs/test/D-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mape_loss-naive-32.yml.slurm
RES84=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-32.yml.slurm)
echo "$RES84, slurm_jobs/test/D-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/D-mape_loss-naive-64.yml.slurm
RES85=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-64.yml.slurm)
echo "$RES85, slurm_jobs/test/D-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mape_loss-naive-128.yml.slurm
RES86=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-128.yml.slurm)
echo "$RES86, slurm_jobs/test/D-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mape_loss-naive-256.yml.slurm
RES87=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-256.yml.slurm)
echo "$RES87, slurm_jobs/test/D-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mape_loss-naive-512.yml.slurm
RES88=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-512.yml.slurm)
echo "$RES88, slurm_jobs/test/D-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/D-mape_loss-naive-1024.yml.slurm
RES89=$(sbatch --parsable slurm_jobs/test/D-mape_loss-naive-1024.yml.slurm)
echo "$RES89, slurm_jobs/test/D-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/H-smape_loss-naive-32.yml.slurm
RES90=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-32.yml.slurm)
echo "$RES90, slurm_jobs/test/H-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-smape_loss-naive-64.yml.slurm
RES91=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-64.yml.slurm)
echo "$RES91, slurm_jobs/test/H-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-smape_loss-naive-128.yml.slurm
RES92=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-128.yml.slurm)
echo "$RES92, slurm_jobs/test/H-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-smape_loss-naive-256.yml.slurm
RES93=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-256.yml.slurm)
echo "$RES93, slurm_jobs/test/H-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-smape_loss-naive-512.yml.slurm
RES94=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-512.yml.slurm)
echo "$RES94, slurm_jobs/test/H-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/H-smape_loss-naive-1024.yml.slurm
RES95=$(sbatch --parsable slurm_jobs/test/H-smape_loss-naive-1024.yml.slurm)
echo "$RES95, slurm_jobs/test/H-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mase_loss-naive-32.yml.slurm
RES96=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-32.yml.slurm)
echo "$RES96, slurm_jobs/test/H-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mase_loss-naive-64.yml.slurm
RES97=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-64.yml.slurm)
echo "$RES97, slurm_jobs/test/H-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mase_loss-naive-128.yml.slurm
RES98=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-128.yml.slurm)
echo "$RES98, slurm_jobs/test/H-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mase_loss-naive-256.yml.slurm
RES99=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-256.yml.slurm)
echo "$RES99, slurm_jobs/test/H-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/H-mase_loss-naive-512.yml.slurm
RES100=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-512.yml.slurm)
echo "$RES100, slurm_jobs/test/H-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mase_loss-naive-1024.yml.slurm
RES101=$(sbatch --parsable slurm_jobs/test/H-mase_loss-naive-1024.yml.slurm)
echo "$RES101, slurm_jobs/test/H-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mape_loss-naive-32.yml.slurm
RES102=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-32.yml.slurm)
echo "$RES102, slurm_jobs/test/H-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mape_loss-naive-64.yml.slurm
RES103=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-64.yml.slurm)
echo "$RES103, slurm_jobs/test/H-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mape_loss-naive-128.yml.slurm
RES104=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-128.yml.slurm)
echo "$RES104, slurm_jobs/test/H-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/H-mape_loss-naive-256.yml.slurm
RES105=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-256.yml.slurm)
echo "$RES105, slurm_jobs/test/H-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mape_loss-naive-512.yml.slurm
RES106=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-512.yml.slurm)
echo "$RES106, slurm_jobs/test/H-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/H-mape_loss-naive-1024.yml.slurm
RES107=$(sbatch --parsable slurm_jobs/test/H-mape_loss-naive-1024.yml.slurm)
echo "$RES107, slurm_jobs/test/H-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-smape_loss-naive-32.yml.slurm
RES108=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-32.yml.slurm)
echo "$RES108, slurm_jobs/test/O-smape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-smape_loss-naive-64.yml.slurm
RES109=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-64.yml.slurm)
echo "$RES109, slurm_jobs/test/O-smape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/O-smape_loss-naive-128.yml.slurm
RES110=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-128.yml.slurm)
echo "$RES110, slurm_jobs/test/O-smape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-smape_loss-naive-256.yml.slurm
RES111=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-256.yml.slurm)
echo "$RES111, slurm_jobs/test/O-smape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-smape_loss-naive-512.yml.slurm
RES112=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-512.yml.slurm)
echo "$RES112, slurm_jobs/test/O-smape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-smape_loss-naive-1024.yml.slurm
RES113=$(sbatch --parsable slurm_jobs/test/O-smape_loss-naive-1024.yml.slurm)
echo "$RES113, slurm_jobs/test/O-smape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mase_loss-naive-32.yml.slurm
RES114=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-32.yml.slurm)
echo "$RES114, slurm_jobs/test/O-mase_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/O-mase_loss-naive-64.yml.slurm
RES115=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-64.yml.slurm)
echo "$RES115, slurm_jobs/test/O-mase_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mase_loss-naive-128.yml.slurm
RES116=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-128.yml.slurm)
echo "$RES116, slurm_jobs/test/O-mase_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mase_loss-naive-256.yml.slurm
RES117=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-256.yml.slurm)
echo "$RES117, slurm_jobs/test/O-mase_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mase_loss-naive-512.yml.slurm
RES118=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-512.yml.slurm)
echo "$RES118, slurm_jobs/test/O-mase_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mase_loss-naive-1024.yml.slurm
RES119=$(sbatch --parsable slurm_jobs/test/O-mase_loss-naive-1024.yml.slurm)
echo "$RES119, slurm_jobs/test/O-mase_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/O-mape_loss-naive-32.yml.slurm
RES120=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-32.yml.slurm)
echo "$RES120, slurm_jobs/test/O-mape_loss-naive-32.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mape_loss-naive-64.yml.slurm
RES121=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-64.yml.slurm)
echo "$RES121, slurm_jobs/test/O-mape_loss-naive-64.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mape_loss-naive-128.yml.slurm
RES122=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-128.yml.slurm)
echo "$RES122, slurm_jobs/test/O-mape_loss-naive-128.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mape_loss-naive-256.yml.slurm
RES123=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-256.yml.slurm)
echo "$RES123, slurm_jobs/test/O-mape_loss-naive-256.yml.slurm" >> submitted_jobs_names.log
chmod u+x slurm_jobs/test/O-mape_loss-naive-512.yml.slurm
RES124=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-512.yml.slurm)
echo "$RES124, slurm_jobs/test/O-mape_loss-naive-512.yml.slurm" >> submitted_jobs_names.log
sleep 5
chmod u+x slurm_jobs/test/O-mape_loss-naive-1024.yml.slurm
RES125=$(sbatch --parsable slurm_jobs/test/O-mape_loss-naive-1024.yml.slurm)
echo "$RES125, slurm_jobs/test/O-mape_loss-naive-1024.yml.slurm" >> submitted_jobs_names.log
