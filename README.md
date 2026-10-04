# CS542200 HW1：Odd-Even Sort（MPI）

繳交程式：`hw1.cc`（編譯：`make` 或 `mpicxx -O3 -o hw1 hw1.cc`，需要課程的 `hash.h` 放在根目錄）

## 實驗（課程機器 Nova，repo 根目錄）

```bash
module load compiler/gcc/13 mpi/openmpi/5.0.10

# 1. 正式計時實驗：big / little / mixed / 2 節點，測資 10，每個設定 5 次
#    單節點的輸入輸出放 node-local /tmp；2 節點（8 process）只能用共享 NFS
bash test/exp/submit_all.sh

# 2. 三個優化的佐證：radix sort、compare-split、active process
sbatch test/opt/run_opt.sh

# 3. Nsight Systems：只看 timeline 與通訊行為，不拿來計時
sbatch test/exp/nsys_job.sh
sbatch --export=ALL,VERSION=ori test/exp/nsys_job.sh

# ---- 全部跑完 ----
python3 test/exp/summarize.py      # → test/exp/results/summary.csv
python3 test/exp/make_report.py    # → test/exp/report/*.png 與 tables.tex
python3 test/exp/nsys_mpi.py test/exp/results/nsys/<資料夾>
python3 test/exp/analyze_input.py /srv/nova/scratch/coursedata/pp2026/hw1/10.in
```

## 檔案

| 路徑 | 用途 |
|---|---|
| `hw1.cc` | 繳交程式 |
| `test/exp/prof_wrap.h` | PMPI 計時外掛（編譯時 `-include`，不改動 hw1.cc） |
| `test/exp/baseline_ori.cc` | 最初版本（前後比較用） |
| `test/exp/submit_all.sh`、`job.sh` | 編譯各版本並送出所有計時實驗 |
| `test/exp/summarize.py` | 整理結果（中位數、speedup；/tmp 與 NFS 分開） |
| `test/exp/make_report.py` | 產生報告的所有圖與 LaTeX 表格 |
| `test/exp/nsys_job.sh`、`nsys_mpi.py` | Nsight Systems profiling 與 timeline 圖 |
| `test/exp/analyze_input.py` | 測資數值分佈 |
| `test/opt/` | 三個優化的 micro-benchmark（`run_opt.sh`） |
| `test/parse_case.py`、`test/gen.py` | 讀測資設定、產生小測資 |
