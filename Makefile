# ---------------------------------------------------------------------------
#  Simple Bitboard Chess Engine -- Makefile
#  Build:   make          (or: mingw32-make on Windows / MinGW)
#  Test:    make test     (perft suite + internal consistency check)
#  Bench:   make bench    (aggregate perft benchmark)
# ---------------------------------------------------------------------------

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
SRC      := main.cpp

ifeq ($(OS),Windows_NT)
  TARGET := chessengine.exe
  RUN    := $(TARGET)
  RM_F   := del /Q
else
  TARGET := chessengine
  RUN    := ./$(TARGET)
  RM_F   := rm -f
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Optional: build with -O3 and native CPU features for extra speed
native: $(SRC)
	$(CXX) $(CXXFLAGS) -O3 -march=native -o $(TARGET) $(SRC)

test: $(TARGET)
	$(RUN) test

check: $(TARGET)
	$(RUN) check

bench: $(TARGET)
	$(RUN) bench

clean:
	-$(RM_F) $(TARGET)

.PHONY: all native test check bench clean
