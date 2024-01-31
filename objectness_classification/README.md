# Objectness Classification

## Contents
```
├── Makefile        -> registers command (eg. log)
├── README.md
├── datasets
│   ├── __init__.py
│   └── ...
├── models
│   ├── lossfn
│   │   ├── __init__.py
│   │   └── ...
│   ├── metrics
│   │   ├── __init__.py
│   │   └── ...
│   ├── __init__.py
│   └── ...
├── datamodule.py   -> looks into `datasets` directory
├── system.py       -> looks into `models` directory
└── train.py        -> uses `datamodule` and `system`
```

## Usage
- To show logs on browser, use the following registered command.
```shell
$ make log
```
