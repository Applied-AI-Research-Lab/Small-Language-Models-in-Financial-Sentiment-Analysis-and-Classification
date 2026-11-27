#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
RES="$HERE/results"; STATE="$RES/_state"; LOGS="$HERE/logs"
GPUS="${GPUS:-0 1 2 3}"
mkdir -p "$RES" "$STATE" "$LOGS"

JOBS=("cot qwen" "supericl_bert qwen" "cot gemma" "supericl_bert gemma"
      "knn qwen" "supericl_svm qwen" "knn gemma" "supericl_svm gemma")

env_setup() {
  cat <<EOF
source "$PROJECT/activate_project.sh" >/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$HERE"
EOF
}

run_fg() {
  bash -c "$(env_setup)
$*"
}

status() {
  echo "== jobs"
  for j in "${JOBS[@]}"; do
    set -- $j; f="$RES/c1/${1}_${2}.csv"; n=0
    [ -f "$f" ] && n=$(( $(wc -l < "$f") - 1 ))
    st="queued"; [ -f "$STATE/${1}_${2}.done" ] && st="DONE"; [ -f "$STATE/${1}_${2}.failed" ] && st="FAILED"
    printf "  %-14s %-6s %-7s rows=%s\n" "$1" "$2" "$st" "$n"
  done
  echo "== tmux"; tmux ls 2>/dev/null | grep '^r2_' || echo "  (no r2_* sessions)"
  [ -f "$RES/ALL_DONE" ] && echo "== ALL DONE -> $RES/summary.md"
}

case "${1:-all}" in
  status) status; exit 0 ;;
  merge)  run_fg "python merge_results.py"; exit $? ;;
  stop)   for s in $(tmux ls 2>/dev/null | cut -d: -f1 | grep '^r2_'); do tmux kill-session -t "$s"; done
          echo "stopped r2_* sessions"; exit 0 ;;
  all)    ;;
  *)      echo "usage: bash run_all.sh [all|status|merge|stop]"; exit 2 ;;
esac

if tmux ls 2>/dev/null | grep -q '^r2_'; then
  echo "r2_* tmux sessions already exist — use 'bash run_all.sh status' or 'bash run_all.sh stop'."; exit 1
fi
FIRST_GPU="${GPUS%% *}"

echo "### 1/4 preflight (environment, files, pinned snapshots)"
run_fg "CUDA_VISIBLE_DEVICES=$FIRST_GPU python preflight.py" 2>&1 | tee "$LOGS/preflight.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: preflight failed (see logs/preflight.log)"; exit 1; }

echo "### 2/4 freeze prompts"
if [ -f "$RES/inputs/manifest.json" ]; then
  echo "prompts already frozen (results/inputs/manifest.json) — not rebuilt"
else
  run_fg "CUDA_VISIBLE_DEVICES=$FIRST_GPU python prep_inputs.py" 2>&1 | tee "$LOGS/prep.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: prep failed (see logs/prep.log)"; exit 1; }
fi

echo "### 3/4 smoke test on GPU $FIRST_GPU (reproduction check + 5 rows per job)"
if [ -f "$STATE/smoke.ok" ]; then
  echo "smoke test already passed — skipped"
else
  run_fg "CUDA_VISIBLE_DEVICES=$FIRST_GPU python preflight.py --repro" 2>&1 | tee "$LOGS/repro.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: reproduction check failed (see logs/repro.log)"; exit 1; }
  for j in "${JOBS[@]}"; do
    set -- $j
    run_fg "CUDA_VISIBLE_DEVICES=$FIRST_GPU python run_job.py --method $1 --slm $2 --smoke" \
      2>&1 | tee "$LOGS/smoke_${1}_${2}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: smoke $1/$2 crashed (see logs/smoke_${1}_${2}.log)"; exit 1; }
  done
  run_fg "python merge_results.py --smoke" 2>&1 | tee "$LOGS/smoke_merge.log"
  run_fg "python - <<'PY'
import pandas as pd, glob, sys
bad = [f for f in glob.glob('results/smoke/*_*.csv') if not f.endswith(('test_round2.csv','metrics_c1.csv'))
       and pd.read_csv(f)['parse_error'].sum() > 1]
print('smoke parse check:', 'FAILED ' + str(bad) if bad else 'OK')
sys.exit(1 if bad else 0)
PY" || { echo "ABORT: too many parse failures in the smoke test"; exit 1; }
  touch "$STATE/smoke.ok"
fi

echo "### 4/4 launching tmux sessions on GPUs: $GPUS"
read -r -a G <<< "$GPUS"
QUEUE=()
for i in "${!JOBS[@]}"; do
  k=$(( i % ${#G[@]} ))
  QUEUE[$k]="${QUEUE[$k]:-}${JOBS[$i]};"
done
for k in "${!G[@]}"; do
  g="${G[$k]}"
  q="${QUEUE[$k]:-}"; [ -z "$q" ] && continue
  script="$LOGS/queue_gpu${g}.sh"
  {
    echo "#!/usr/bin/env bash"
    env_setup
    echo "export CUDA_VISIBLE_DEVICES=$g"
    IFS=';' read -r -a items <<< "$q"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      set -- $it
      cat <<EOF
if [ ! -f "$STATE/${1}_${2}.done" ]; then
  rm -f "$STATE/${1}_${2}.failed"
  ok=0
  for attempt in 1 2; do
    echo "[\$(date '+%F %T')] GPU $g: $1/$2 attempt \$attempt" | tee -a "$LOGS/${1}_${2}.log"
    python run_job.py --method $1 --slm $2 2>&1 | tee -a "$LOGS/${1}_${2}.log"
    if [ "\${PIPESTATUS[0]}" -eq 0 ]; then ok=1; break; fi
    sleep 30
  done
  if [ \$ok -eq 1 ]; then touch "$STATE/${1}_${2}.done"; else touch "$STATE/${1}_${2}.failed"; fi
fi
EOF
    done
    echo "echo \"[\$(date '+%F %T')] GPU $g queue finished\""
  } > "$script"
  tmux new-session -d -s "r2_gpu${g}" "bash '$script'; exec bash"
  echo "  r2_gpu${g}: ${q//;/ | }"
done

watch="$LOGS/watch_merge.sh"
{
  echo "#!/usr/bin/env bash"
  env_setup
  echo "while true; do"
  echo "  n=\$(ls \"$STATE\"/*.done \"$STATE\"/*.failed 2>/dev/null | grep -v smoke | wc -l)"
  echo "  [ \"\$n\" -ge ${#JOBS[@]} ] && break; sleep 300"
  echo "done"
  echo "python merge_results.py 2>&1 | tee \"$LOGS/merge.log\""
  echo "touch \"$RES/ALL_DONE\"; echo ALL DONE"
} > "$watch"
tmux new-session -d -s r2_merge "bash '$watch'; exec bash"

echo
echo "Launched. Follow with:  bash run_all.sh status   |   tmux attach -t r2_gpu${G[0]}"
echo "When finished: results/summary.md, results/metrics_c1.csv, results/mcnemar_c1.json"
