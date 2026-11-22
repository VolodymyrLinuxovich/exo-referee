# Builds the shared library loaded by exo/fastci.py.
UNAME := $(shell uname)
EXT := $(if $(filter Darwin,$(UNAME)),dylib,so)
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC

libfastci.$(EXT): fastci.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean:
	rm -f libfastci.dylib libfastci.so
.PHONY: clean
