**Instructions**
目前在after modify版本之后可以正常跑了，但是仍然无法收敛。
尝试帧堆叠，并放在frame_stuck_version下。
loss, perf = agent.update(trajectory) 改stuck_frame 报错是next_state的信息没有使用堆叠的帧信息
修改之后看起来训练有效果了
按照TODO添加功能，进一步分析，调优
# todo list:加可视化 加batch 支持npu 加MHA