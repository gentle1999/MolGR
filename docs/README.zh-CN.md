# MolGR 文档

[English](README.md) | [中文](README.zh-CN.md)

本目录保存根 README 之外的项目文档。

## 快速入口

- [`../README.md`](../README.md)：英文项目概览、安装、开发命令、测试和 benchmark 快速开始。
- [`../README.zh-CN.md`](../README.zh-CN.md)：中文项目 README。
- [`../benchmarks/README.md`](../benchmarks/README.md)：英文 benchmark 文档。
- [`../benchmarks/README.zh-CN.md`](../benchmarks/README.zh-CN.md)：中文 benchmark 环境、共享方法列表、输入格式和输出 schema。

## 架构文档

- [`architecture/ALGORITHM_ARCHITECTURE.md`](architecture/ALGORITHM_ARCHITECTURE.md)：英文算法架构说明。
- [`architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](architecture/ALGORITHM_ARCHITECTURE.zh-CN.md)：中文算法架构说明。
- [`architecture/ALGORITHM_ARCHITECTURE.svg`](architecture/ALGORITHM_ARCHITECTURE.svg)：紧凑的出版物级算法架构图。
- [`architecture/ALGORITHM_ARCHITECTURE.drawio`](architecture/ALGORITHM_ARCHITECTURE.drawio)：架构图可编辑源文件。

这些文档覆盖公开入口 `xyz_to_rdmol(...)`、Python fallback 参考后端、C++ `_core`
后端、金属感知重建、共振恢复、评分策略、C++ 优化和维护边界。

## 发布文档

- [`release/DEVELOPMENT_RELEASE_GUIDE.md`](release/DEVELOPMENT_RELEASE_GUIDE.md)：英文开发与发布指南。
- [`release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md`](release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md)：中文开发与发布指南。

这些文档说明 Gitea 内网包通道、GitHub/PyPI 正式发布路径、wheel 构建矩阵、标签策略和
必要 CI 变量。

## 开发工具

- [`development/MOLECULE_REVIEW_TOOL.zh-CN.md`](development/MOLECULE_REVIEW_TOOL.zh-CN.md)：MolGR 重建审核工具的运行环境、队列、审核状态与 fixture 流程。
- [`development/MOLECULE_REVIEW_TOOL.md`](development/MOLECULE_REVIEW_TOOL.md)：英文开发工具说明。
