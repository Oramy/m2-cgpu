CXX = g++
CXXFLAGS = -O2 -Wall
SRCS = $(wildcard *.cxx)
PROG = boids
OBJS = $(SRCS:%.cxx=%.o)
LDFLAGS = -lm  -o $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS)

%.o:%.cxx
	$(CXX) -c $(CXXFLAGS) $<

test: $(PROG)
	./$(PROG)

clean:
	rm -f $(OBJS) $(PROG) *~
