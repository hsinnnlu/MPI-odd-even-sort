CXX      = mpicxx
CXXFLAGS = -O3

# 和作業規定的編譯方式相同（mpicxx -O3 -o hw1 hw1.cc）
hw1: hw1.cc hash.h
	$(CXX) $(CXXFLAGS) -o $@ hw1.cc

clean:
	rm -f hw1

.PHONY: clean
