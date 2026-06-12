题目是 C_20260325_中文翻译.md
数据说明/测试方法见 README.md
题意补充 当软硬约束可能存在冲突的时候 必须满足硬约束的要求


当我要求你完成解题任务时
你需要将做法写在.\sol下的某个文件夹内（我询问的时候会指定）
首先设计做法在sol.md中 之后等检查确认后 等我指令写代码 
代码完成后用 iccad2026_evaluate.py 进行测试（测试命令 python iccad2026_evaluate.py --evaluate my_optimizer.py 注意记得调整目录），并将结果输出到 result.txt中（和代码同一个目录下面）
其他sol文件你可以参考其接口 但不要参考其做法。
之后用 python analyze_cost_contributions.py --evaluate 【需要测试的代码】 进行结果分析 输出到 analyze.txt中（和代码同一个目录下面）