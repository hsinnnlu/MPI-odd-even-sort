CXX      = mpicxx
CXXFLAGS = -O3

# 正式版：和作業規定的編譯方式相同（mpicxx -O3 -o hw1 hw1.cc）
hw1: hw1.cc hash.h
	$(CXX) $(CXXFLAGS) -o $@ hw1.cc

# 單機參考答案
test/ref: test/ref.cc hash.h
	g++ -O2 -I. -o $@ test/ref.cc

# 跨節點 MPI 連線測試（和 hw1 無關）
test/mpi_ping: test/mpi_ping.cc
	$(CXX) -O2 -o $@ test/mpi_ping.cc

all: hw1 test/ref test/mpi_ping

clean:
	rm -f hw1 test/ref test/mpi_ping

.PHONY: all clean
