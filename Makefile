BUILD_DIR := build
EXE3D     := $(BUILD_DIR)/mat_free_mass3D


profile3D:
	ncu -f -o mat_free_mass3D.ncu-rep --set full $(EXE3D) --nelements 20 --nreps 1
