# Codex 启动提示词

在一个全新的 Git 仓库中，严格执行 `2026-08-05-tristatelite-phase1-implementation-plan.md`，使用测试驱动开发和小步提交。

本轮只完成 Phase 1：

1. 初始化 Python 3.11 项目；
2. 从 NASA 官方资源下载 `battery_alt_dataset.zip`；
3. 记录文件大小、SHA256、来源 URL 和下载时间；
4. 安全盘点 ZIP 内全部文件；
5. 探测文本/CSV/MAT 的候选字段与变量；
6. 生成 `nasa_archive_inventory.json`、`nasa_schema_report.json` 和 `nasa_schema_report.md`；
7. 运行所有测试和 lint；
8. 提交代码；
9. 停止。

禁止实现标签、数据划分、窗口、模型或训练。禁止在真实 schema report 生成前猜测 battery ID、cycle ID、time、voltage、current、temperature、phase、capacity 或文件格式。

最终回复必须包含：

- 仓库目录树；
- 每个命令的退出状态；
- 测试和 lint 结果；
- ZIP 大小与 SHA256；
- 文件扩展名统计；
- top-level 目录统计；
- 最高排名的 5 个候选时序文件；
- 候选字段/变量及其证据；
- 需要人工确认的所有歧义；
- 最后明确写出“已在 Phase 1 gate 停止，尚未开始模型实现”。
