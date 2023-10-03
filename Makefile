.PHONY: gpu_setting

gpu_setting:
	poetry remove torch torchvision
	poetry source add torch_cu118 --priority=explicit https://download.pytorch.org/whl/cu118
	poetry add torch torchvision --source torch_cu118
