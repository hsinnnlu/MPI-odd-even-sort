CXX      = mpicxx
CXXFLAGS = -O3

# 正式版：和作業規定的編譯方式相同（mpicxx -O3 -o hw1 hw1.cc）
hw1: hw1.cc hash.h
	$(CXX) $(CXXFLAGS) -o $@ hw1.cc

# 有計時輸出的版本（stderr 會印出 PROFILE ...），給報告用
hw1_profile: hw1.cc hash.h
	$(CXX) $(CXXFLAGS) -DHW1_PROFILE -o $@ hw1.cc

# 單機參考答案
test/ref: test/ref.cc hash.h
	g++ -O2 -I. -o $@ test/ref.cc

all: hw1 hw1_profile test/ref

clean:
	rm -f hw1 hw1_profile test/ref

.PHONY: all clean
