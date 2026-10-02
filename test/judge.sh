#!/bin/bash
# =============================================================================
# 仿 hw1-judge：一個指令完成「編譯 → 檢查跨節點 MPI → 跑全部公開測資 → 比對 → 計時」
#
# 用法（在 repo 根目錄或任何地方都可以）：
#   bash test/judge.sh                 # 跑全部測資
#   bash test/judge.sh 09 19 36        # 只跑指定編號
#   REPEAT=5 bash test/judge.sh        # 每筆跑 5 次，時間取中位數（報告用）
#
# 可用環境變數：
#   CASES     測資目錄（預設 /srv/nova/scratch/coursedata/pp2026/hw1）
#   HASH_H    hash.h 的位置（預設自動在課程目錄裡找）
#   REPEAT    每筆測資重複次數（預設 1）
#   TIMEOUT   每次執行的時間上限，秒（預設 300）
#   NO_BUILD=1  不重新編譯，直接用現有的 ./hw1
#
# 結果：
#   螢幕上印出每筆的對錯與時間
#   test/out/judge_<時間>.csv  每次執行的詳細紀錄（可直接拿來畫圖）
# =============================================================================
set -u
cd "$(dirname "$0")/.."

CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
REPEAT=${REPEAT:-1}
TIMEOUT=${TIMEOUT:-300}
# 輸出一定要放在所有節點都看得到的地方（NFS），不能用 /tmp
OUT_DIR=${OUT_DIR:-$PWD/test/out}
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUT_DIR/judge_$STAMP.csv"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*"; exit 1; }

HAVE_SRUN=0
command -v srun >/dev/null 2>&1 && HAVE_SRUN=1

# -----------------------------------------------------------------------------
# 1. 載入作業指定的 module（非互動 shell 預設沒有 module 指令，要先 source）
# -----------------------------------------------------------------------------
say "== [1/4] 環境"
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi
if type module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1
    module load compiler/gcc/13 mpi/openmpi/5.0.10 2>/dev/null \
        || warn "無法載入 compiler/gcc/13 mpi/openmpi/5.0.10，改用目前環境"
else
    warn "找不到 module 指令，使用目前環境的編譯器"
fi
echo "mpicxx: $(command -v mpicxx || echo '找不到')"
mpicxx --version 2>/dev/null | head -1

# -----------------------------------------------------------------------------
# 2. 準備 hash.h 並編譯（和作業規定相同：mpicxx -O3 -o hw1 hw1.cc）
# -----------------------------------------------------------------------------
say "== [2/4] 編譯"
if [ ! -f hash.h ]; then
    src=${HASH_H:-$(find /srv/nova/projects/courses/pp2026/hw1 /srv/nova/scratch/coursedata/pp2026/hw1 \
                         -maxdepth 3 -name hash.h 2>/dev/null | head -1)}
    [ -n "$src" ] && [ -f "$src" ] || die "找不到 hash.h，請用 HASH_H=/path/to/hash.h 指定"
    cp "$src" hash.h && echo "已複製 hash.h：$src"
fi
if [ "${NO_BUILD:-0}" != 1 ]; then
    rm -f hw1
    mpicxx -O3 -o hw1 hw1.cc || die "hw1.cc 編譯失敗"
    mpicxx -O2 -o test/mpi_ping test/mpi_ping.cc || die "test/mpi_ping.cc 編譯失敗"
    echo "編譯完成"
fi
[ -x ./hw1 ] || die "找不到 ./hw1"

# -----------------------------------------------------------------------------
# 3. 檢查跨節點 MPI 通訊
#    已知問題：節點上若有「每台 IP 都一樣」的網卡（例如 docker0、virbr0），
#    Open MPI 的 TCP 連線會連錯人（received unexpected process identifier）。
#    這裡先測一次；失敗的話，自動找出這些網卡並排除，再測一次。
# -----------------------------------------------------------------------------
say "== [3/4] 跨節點 MPI 連線檢查"
MULTI_NODE_NOTE=""
if [ $HAVE_SRUN = 1 ]; then
    ping_ok() { timeout 60 srun -p big -N2 -n8 ./test/mpi_ping 2>&1 | grep -q "mpi_ping OK"; }

    if ping_ok; then
        echo "跨節點通訊正常"
    else
        warn "跨節點通訊失敗，嘗試排除重複 IP 的網卡……"
        # 每個節點列出自己的 IPv4（-l 會在每行前面加 task 編號）。
        # 同一個 IP 出現在兩台節點上 → 那張網卡不能用來跨節點連線。
        bad_ifaces=$(srun -p big -N2 -n2 -l ip -4 -o addr show 2>/dev/null | awk '
            $3 != "lo" { iface = $3; split($5, a, "/"); ip = a[1]; seen[ip]++; name[ip] = iface }
            END { for (ip in seen) if (seen[ip] > 1) print name[ip] }' | sort -u | tr '\n' ',')
        export OMPI_MCA_btl_tcp_if_exclude="lo,${bad_ifaces%,}"
        export OMPI_MCA_btl_tcp_if_exclude="${OMPI_MCA_btl_tcp_if_exclude%,}"
        echo "OMPI_MCA_btl_tcp_if_exclude=$OMPI_MCA_btl_tcp_if_exclude"

        if ping_ok; then
            echo "排除後跨節點通訊正常"
            MULTI_NODE_NOTE="注意：多節點測資是在 OMPI_MCA_btl_tcp_if_exclude=$OMPI_MCA_btl_tcp_if_exclude 下通過的。
      這是叢集網路設定的問題（和 hw1.cc 無關），建議回報 TA，正式 judge 環境需要修正。"
        else
            unset OMPI_MCA_btl_tcp_if_exclude
            MULTI_NODE_NOTE="注意：跨節點 MPI 通訊無法建立，多節點測資的失敗和 hw1.cc 無關，請回報 TA。
      可以用 srun -p big -N2 -n8 ./test/mpi_ping 重現。"
            warn "排除網卡後仍然失敗，多節點測資預期會失敗"
        fi
    fi
else
    warn "沒有 srun，改用 mpirun 在本機執行（忽略 partition 與節點數）"
fi

# -----------------------------------------------------------------------------
# 4. 跑測資
# -----------------------------------------------------------------------------
say "== [4/4] 測資（每筆 $REPEAT 次）"
if [ $# -gt 0 ]; then ids=("$@"); else
    ids=(); for f in "$CASES"/*.txt; do ids+=("$(basename "$f" .txt)"); done
fi
[ ${#ids[@]} -gt 0 ] || die "在 $CASES 找不到測資"

echo "case,n,nodes,procs,partition,rounds,trial,seconds,result" > "$CSV"
printf "%-5s %-10s %-5s %-5s %-7s %-9s %s\n" case N nodes procs part "time(s)" result
pass=0; fail=0; total_time=0

for id in "${ids[@]}"; do
    if ! read -r n nodes procs part rounds < <(python3 test/parse_case.py "$CASES/$id.txt" 2>/dev/null); then
        printf "%-5s 無法解析 %s\n" "$id" "$CASES/$id.txt"; fail=$((fail+1)); continue
    fi
    out="$OUT_DIR/$id.out"
    log="$OUT_DIR/$id.log"
    times=(); result="OK"

    for ((t = 1; t <= REPEAT; ++t)); do
        head -c 4096 /dev/urandom > "$out"   # 先放舊資料，檢查程式有沒有截斷
        if [ $HAVE_SRUN = 1 ]; then
            cmd=(srun -p "$part" -N"$nodes" -n"$procs" ./hw1 "$n" "$CASES/$id.in" "$out" "$rounds")
        else
            cmd=(mpirun ${MPIRUN_ARGS:-} -np "$procs" ./hw1 "$n" "$CASES/$id.in" "$out" "$rounds")
        fi
        start=$(date +%s.%N)
        timeout "$TIMEOUT" "${cmd[@]}" > "$log" 2>&1
        rc=$?
        sec=$(python3 -c "print(f'{$(date +%s.%N) - $start:.3f}')")

        if [ $rc -eq 124 ]; then r="TIMEOUT"
        elif [ $rc -ne 0 ]; then r="FAIL(exit $rc)"
        elif msg=$(python3 test/fcmp.py "$CASES/$id.out" "$out"); then r="OK"
        else r="WRONG"; echo "  $id: $msg" >> "$log"
        fi
        echo "$id,$n,$nodes,$procs,$part,$rounds,$t,$sec,$r" >> "$CSV"
        times+=("$sec")
        [ "$r" = OK ] || result="$r"
    done

    med=$(printf '%s\n' "${times[@]}" | python3 -c "import sys,statistics as s; print(f'{s.median(float(x) for x in sys.stdin):.3f}')")
    total_time=$(python3 -c "print(f'{$total_time + $med:.3f}')")
    if [ "$result" = OK ]; then pass=$((pass+1)); shown="OK"
    else fail=$((fail+1)); shown="$result（見 $log）"; fi
    printf "%-5s %-10s %-5s %-5s %-7s %-9s %s\n" "$id" "$n" "$nodes" "$procs" "$part" "$med" "$shown"
done

echo
say "通過 $pass / $((pass + fail))，總時間（各筆中位數相加）$total_time 秒"
echo "詳細紀錄：$CSV"
[ -n "$MULTI_NODE_NOTE" ] && warn "$MULTI_NODE_NOTE"
command -v hw1-judge >/dev/null 2>&1 && echo "（偵測到官方 hw1-judge，可另外執行它做最終確認）"
[ $fail -eq 0 ]
