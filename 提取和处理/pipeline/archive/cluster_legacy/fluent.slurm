#!/bin/bash
#Author: DJ
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH -w node01
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=Fluent
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#export ANSYSLMD_LICENSE_FILE=1055@master
#export ANSYSLI_SERVERS=2325@master
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
cd $SLURM_SUBMIT_DIR
JOU=./2.jou
time fluent 3ddp -g -t64 -mpi=intel -platform=intel -i $JOU  -ssh

# 1. 清理前 1119 个产生的 ASCII (CSV) 文件（保留 1120）
echo "Starting cleanup of intermediate ASCII files (1-1119)..."
for i in $(seq -f "%04g" 1 1119); do
    # 使用通配符匹配任何前缀（人名），只要后缀是 -$i
    rm -f *-$i
done

# 2. 1121 至 1280 隔一个删除一个（删除 1121, 1123...，保留 1120, 1122, 1124...）
echo "Starting interval cleanup (1121-1280, keeping every other file)..."
for i in $(seq 1121 2 1280); do
    suffix=$(printf "%04g" $i)
    rm -f *-$suffix
done
echo "Cleanup complete. Data files (.dat/.cas) and selected monitoring files have been preserved."

# 3. 清理所有 .trn 类型临时文件
echo "Cleaning up .trn files..."
rm -f *.trn 2>/dev/null


# 4. 将剩余带数字后缀的文件归档到 ascii_in 文件夹
echo "Moving remaining numbered files to ascii_in folder..."
mkdir -p ascii_in
mv *-[0-9][0-9][0-9][0-9] ./ascii_in/ 2>/dev/null

# 5. 将所有 .out 文件类型保存到 Global_conditions
echo "Moving monitoring files (.out) to Global_conditions folder..."
mkdir -p Global_conditions
mv *.out ./Global_conditions/ 2>/dev/null

# 6. 进入 ascii 文件夹删掉从 1 到 1119 号的文件，以及 1121-1280 隔一个删除一个
echo "Cleaning up intermediate files inside ascii folder..."
if [ -d "ascii" ]; then
    cd ascii
    # 删除 1-1119
    for i in $(seq -f "%04g" 1 1119); do
        rm -f *-$i
    done
    # 删除 1121-1280 (隔一个删一个)
    for i in $(seq 1121 2 1280); do
        suffix=$(printf "%04g" $i)
        rm -f *-$suffix
    done
    cd ..
fi

echo "Process finished."

