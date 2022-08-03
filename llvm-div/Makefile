LLVM_DIR=/usr/local/Cellar/llvm@12/12.0.1_1/bin/

div:
	$(LLVM_DIR)/llvm-as div.ll
	$(LLVM_DIR)/llc div.bc
	as div.s -o div.o
	ar crs libdiv.a div.o
	rustc main.rs -O -g -L .
