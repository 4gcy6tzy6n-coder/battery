# SOTA 后处理剧本（cron `ad2c02f6` 触发时执行）

> 当 `data/experiments/sota-v2-s{0,1}/results.json` 两边都落地时按本剧本执行。
> 触发条件：`test -f data/experiments/sota-v2-s0/results.json && test -f data/experiments/sota-v2-s1/results.json`

## 0. 验证（先做）

```bash
cd /Users/yyl/Desktop/workshop/battery

# 进程应已结束
ps -p 85694,85695 2>/dev/null || echo "training processes gone"

# 结果完整性
ls -la data/experiments/sota-v2-s0/ data/experiments/sota-v2-s1/
# 期望：每个目录都有 best_model.pt / results.json / test_predictions.npz
```

## 1. 重校准（per run）

```bash
for seed in 0 1; do
  out="data/experiments/sota-v2-s$seed"
  if [ -f "$out/results.json" ] && [ ! -f "$out/recalibrated_results.json" ]; then
    .venv/bin/python scripts/recalibrate_run.py \
      --run "$out" \
      --config configs/experiments/sota_unordered_nophysics.yaml
  fi
done
```

## 2. 独立复核（核心数字）

```bash
.venv/bin/python scripts/verify_results.py \
  --runs-dir data/experiments \
  --include sota-v2-s0,sota-v2-s1 \
  --write data/experiments/runtime_capture/verify_sota.json
```

## 3. 聚合

```bash
# 把 sota-v2-s0/s1 也归到标准 runs/ 布局以便 aggregate 统一处理
# 选项 A：在 data/experiments/runs/ 下加软链（不推荐，aggregate 扫目录）
# 选项 B：直接给 aggregate 一个含 sota-v2-s0/s1 子树的临时根
mkdir -p data/experiments/runs_sota
ln -sfn ../../sota-v2-s0 data/experiments/runs_sota/sota_unordered_nophysics/seed0
ln -sfn ../../sota-v2-s1 data/experiments/runs_sota/sota_unordered_nophysics/seed1

.venv/bin/python scripts/aggregate_results.py \
  --root data/experiments/runs \
  --format markdown > data/experiments/runtime_capture/aggregate_main.md
.venv/bin/python scripts/aggregate_results.py \
  --root data/experiments/runs_sota \
  --format markdown > data/experiments/runtime_capture/aggregate_sota.md
```

## 4. 更新论文结果段

```bash
# 把聚合表追加到 docs/research/paper_results_draft.md
# 比较 unordered(基线) vs sota_unordered_nophysics(新)：soc/soh/log_tte/tte_seconds CRPS/MAE/PICP
# 检查 elapsed_s 特征是否带来一致改进
```

## 5. 更新 progress_log

在 `docs/research/progress_log.md` 末尾追加：
- 完成时间、wall 时长、最终 epoch 数、test 关键指标
- 与 baseline `ablation_unordered` 的对比数字
- 是否触发了 early stop

## 6. 提交并推 40 commit

```bash
git add docs/ scripts/capture_sota_runtime.sh scripts/watch_sota_results.sh docs/research/sota_postprocess.md docs/research/progress_log.md
git commit -m "feat: SOTA s0/s1 complete; recalibrate; update results draft

- sota_unordered_nophysics seed 0/1 训练完成于 data/experiments/sota-v2-s{0,1}
- 触发 docs/research/sota_postprocess.md 后处理剧本
- 更新 docs/research/paper_results_draft.md 与 progress_log

Co-Authored-By: Claude <noreply@anthropic.com>"

git push -u origin agent/phase2b-probabilistic-core
```

## 7. Goal 解除

完成后即可 `/goal clear`（cron job `ad2c02f6` 用 `CronDelete ad2c02f6` 取消，避免下次空闲再触发）。